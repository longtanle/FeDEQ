import os
import functools
from functools import partial
import pickle

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers
import haiku as hk

import optax

from utils import jax_utils
from  utils import model_utils
from utils import data_utils
from utils.data_utils import gen_batch, client_selection

from trainers.base import BaseTrainerLocal


class Ditto_ResNet(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running Ditto')

    # Set loss function and model. For now, fixed model arch for every task.
    if self.dataset in ('femnist', 'cifar10', 'cifar100'):
      self.data_loss_fn = jax_utils.sce_loss_hk_resnet
      self.pred_fn = jax_utils.multiclass_classify_resnet
      self.model_fn, \
        self.mode_prefix, \
          self.last_layers = model_utils.get_model(self.model, 
                                                  self.dataset,
                                                  self.fwd_solver,
                                                  self.bwd_solver)

    else:
      raise ValueError(f'Unsupported dataset: {self.dataset}')

    # Create model architecture & compile prediction/loss function
    self.model  = hk.transform_with_state(self.model_fn)
    #self.model = hk.without_apply_rng(hk.transform(self.model_fn))
    self.pred_fn = jit(functools.partial(self.pred_fn, self.model))
    self.data_loss_fn = jit(functools.partial(self.data_loss_fn, self.model))

    # Since Ditto take multiple inner iters, we overwrite noise & data generator
    self.global_iters = 1
    self.local_iters = 1
    step_factor = self.global_iters + self.local_iters
    #self.noise_mults = dp_accounting.compute_silo_noise_mults(self.num_clients,
    #                                                          self.train_samples,
    #                                                          args,
    #                                                          steps_factor=step_factor)
    for t in range(self.num_clients):
      self.batch_gen[t] = gen_batch(self.x_train[t],
                                    self.y_train[t],
                                    self.batch_sizes[t],
                                    num_iter=step_factor * (self.num_rounds + 1) * self.inner_iters)

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    global_params, net_state = self.model.init(key, data_batch, is_training = True)

    num_params = hk.data_structures.tree_size(global_params)
    byte_size = hk.data_structures.tree_bytes(global_params)

    print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')

    local_params = []
    local_net_states = []
    for t in range(self.num_clients):
      local_param, local_net_state = self.model.init(key, data_batch, is_training = True)
      local_params.append(local_param)
      local_net_states.append(local_net_state)

    local_global_updates = [0] * self.num_clients
    net_state_updates = [0] * self.num_clients

    # Optimizer shared for every client (re-init before client work)
    opt = optax.sgd(self.lr)  # provides init_fn, update_fn, params_fn

    def loss_fn(params, net_state, rng, batch, global_params):
      mean_batch_term, net_state = self.data_loss_fn(params, net_state, rng, batch)
      flat_params = jax_utils.model_flatten(params)
      model_diff = flat_params - jax_utils.model_flatten(global_params)
      prox_term = 0.5 * self.lam * (model_diff @ model_diff)
      l2_term = 0.5 * self.l2_reg * (flat_params @ flat_params)
      return mean_batch_term + prox_term + l2_term, net_state

    def batch_update(key, params, batch_idx, opt_state, net_state, global_params, batch):
      key = random.fold_in(key, batch_idx)
      #params = opt.params_fn(opt_state)
      grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
      (loss, net_state), mean_grad = grad_fn(params, net_state, key, batch, global_params)
      updates, opt_state = opt.update(mean_grad, opt_state)
      params = optax.apply_updates(params, updates)
      #mean_grad = grad(loss_fn)(params, batch, global_params)
      #return key, opt.update_fn(batch_idx, mean_grad, opt_state)
      return key, params, opt_state, net_state

    batch_update = jit(batch_update)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[Ditto] Round',
                  disable=(self.args['repeat'] != 1)):
      key = random.fold_in(key, i)
      chosen_idx = np.random.choice(self.num_clients, 
                                    replace=False,
                                    size=self.num_clients_per_round)
        
      selected_clients = list(range(self.num_clients))  # NOTE: use all clients for cross-silo.
      for t in chosen_idx:
        key = random.fold_in(key, t)
        # Batch generator
        if self.inner_mode == 'iter':
          global_batches = (next(self.batch_gen[t])
                            for _ in range(self.inner_iters * self.global_iters))
          local_batches = (next(self.batch_gen[t])
                           for _ in range(self.inner_iters * self.local_iters))
        else:
          epoch_gen_fn = functools.partial(data_utils.epochs_generator, 
                                           self.x_train[t], 
                                           self.y_train[t],
                                           self.batch_sizes[t])
          global_batches = epoch_gen_fn(epochs=self.inner_epochs * self.global_iters,
                                        seed=int(key[0]))
          local_batches = epoch_gen_fn(epochs=self.inner_epochs * self.local_iters,
                                       seed=int(key[1]))

        # Global updates
        #opt_state = opt.init_fn(global_params)
        params = global_params
        opt_state = opt.init(global_params)
        net_state_t= net_state

        for batch_idx, batch in enumerate(global_batches):
          prox_params = params
          key, params, opt_state, local_net_states[t] = batch_update(key,
                                                                      params,
                                                                      batch_idx, 
                                                                      opt_state,
                                                                      local_net_states[t], 
                                                                      prox_params, 
                                                                      batch)
          
        new_global_params = params
        # Local updates
        opt_state = opt.init(local_params[t])
        for batch_idx, batch in enumerate(local_batches):
          prox_params = global_params
          key, params, opt_state, local_net_states[t] = batch_update(key,
                                                                    params,
                                                                    batch_idx,
                                                                    opt_state,
                                                                    local_net_states[t],
                                                                    prox_params,
                                                                    batch)
                                                
        new_local_params = params

        # Record new *local* model and *global* model diff
        local_global_updates[t] = jax_utils.model_subtract(new_global_params, global_params)
        net_state_updates[t] = jax_utils.model_subtract(local_net_states[t], net_state)

        local_params[t] = new_local_params

      # Update global model
      round_local_updates = [local_global_updates[idx] for idx in chosen_idx]
      round_weight_updates = np.asarray([self.update_weights[idx] for idx in chosen_idx])
      round_net_state_updates = [net_state_updates[idx] for idx in chosen_idx]

      average_update = jax_utils.model_average(round_local_updates, weights=round_weight_updates)

      average_net_state_update = jax_utils.model_average(round_net_state_updates,
                                                         weights=round_weight_updates)
           
      global_params = jax_utils.model_add(global_params, average_update)
      net_state = jax_utils.model_add(net_state, average_net_state_update)

      local_global_updates = [0] * self.num_clients
      net_state_updates = [0] * self.num_clients
      
      if i % self.args['eval_every'] == 0:
        train_accu, test_accu = self.eval_resnet(local_params, local_net_states, i)

    pickle.dump(global_params, open(os.path.join(self.args['outdir'], "model.pkl"), "wb"))

    if self.num_unseen != 0:
      train_accu_unseen, test_accu_unseen = self.eval_unseen(global_params, net_state)   

    return train_accu, test_accu

  def eval_resnet(self, local_params, net_states, round_idx):

    #local_params = [params] * self.num_clients
    #net_states = [net_state] * self.num_clients
    key = random.PRNGKey(self.seed)

    train_losses, test_losses = [], []
    num_correct_train, num_correct_test = [], []
    for t in range(self.num_clients):
      key = random.fold_in(key, t)      
      # Train
      train_preds, _ = self.pred_fn(local_params[t], net_states[t], key, self.x_train[t])
      num_correct_train.append(jnp.sum(train_preds == self.y_train[t]))
      train_loss, _ = self.data_loss_fn(local_params[t], net_states[t], key, (self.x_train[t], self.y_train[t]))
      train_losses.append(train_loss)
      # Test
      test_preds, _ = self.pred_fn(local_params[t], net_states[t], key, self.x_test[t])
      num_correct_test.append(jnp.sum(test_preds == self.y_test[t]))
      test_loss, _  = self.data_loss_fn(local_params[t], net_states[t], key,(self.x_test[t], self.y_test[t]))
      test_losses.append(test_loss)

    avg_train_metric = np.sum(np.array(num_correct_train)) / np.sum(self.train_samples)
    avg_test_metric = np.sum(np.array(num_correct_test)) / np.sum(self.test_samples)
    avg_train_loss = np.average(train_losses, weights=self.train_samples)
    avg_test_loss = np.average(test_losses, weights=self.test_samples)

    if not self.args['quiet']:
      print(f'Round {round_idx}, avg train metric: {avg_train_metric:.5f},'
            f'avg test metric: {avg_test_metric:.5f}')

    # Save only 5 decimal places
    data_utils.print_log(np.round([avg_train_metric, avg_test_metric], 5).tolist(),
                    stdout=False,
                    fpath=os.path.join(self.args['outdir'], 'output.txt'))
    if not self.args['is_regression']:
      data_utils.print_log(np.round([avg_train_loss, avg_test_loss], 5).tolist(),
                      stdout=False,
                      fpath=os.path.join(self.args['outdir'], 'losses.txt'))

    return avg_train_metric, avg_test_metric
  

  def eval_unseen(self, params, net_state):

    # NOTE: use all unseen clients for personalization
    unseen_params = [params] * self.num_unseen
    unseen_net_states = [net_state] * self.num_unseen
    key = random.PRNGKey(self.seed)

    unseen_train_losses, unseen_test_losses = [], []
    num_correct_train, num_correct_test = [], []
    for t in range(self.num_unseen):
      key = random.fold_in(key, t)      
      # Train
      train_preds, _ = self.pred_fn(unseen_params[t], unseen_net_states[t], key, self.x_unseen_train[t])
      num_correct_train.append(jnp.sum(train_preds == self.y_unseen_train[t]))
      #train_loss, _ = self.data_loss_fn(t_params, local_net_states[t], key, (self.x_train[t], self.y_train[t]))
      #train_losses.append(train_loss)
      # Test
      test_preds, _ = self.pred_fn(unseen_params[t], unseen_net_states[t], key, self.x_unseen_test[t])
      num_correct_test.append(jnp.sum(test_preds == self.y_unseen_test[t]))
      #test_loss, _  = self.data_loss_fn(t_params, local_net_states[t], key,(self.x_test[t], self.y_test[t]))
      #test_losses.append(test_loss)

    avg_unseen_train_metric = np.average(num_correct_train, weights=self.unseen_train_samples)
    avg_unseen_test_metric = np.average(num_correct_test, weights=self.unseen_test_samples)

    avg_train_metric = np.sum(np.array(num_correct_train)) / np.sum(self.unseen_train_samples)
    avg_test_metric = np.sum(np.array(num_correct_test)) / np.sum(self.unseen_test_samples)
    
    #avg_train_loss = np.average(train_losses, weights=self.train_samples)
    #avg_test_loss = np.average(test_losses, weights=self.test_samples)

    if not self.args['quiet']:
      print(f'[Generalization], avg unseen train metric: {avg_unseen_train_metric:.5f},'
            f'avg unseen test metric: {avg_unseen_test_metric:.5f}',
            f'avg train metric: {avg_train_metric:.5f}',
            f'avg test metric: {avg_test_metric:.5f}')

    # Save only 5 decimal places
    data_utils.print_log(np.round([avg_unseen_train_metric, avg_unseen_test_metric], 5).tolist(),
                    stdout=False,
                    fpath=os.path.join(self.args['outdir'], 'unseen.txt'))

    return avg_unseen_train_metric, avg_unseen_test_metric    
