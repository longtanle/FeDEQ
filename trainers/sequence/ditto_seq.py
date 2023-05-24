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


class DittoSEQ(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running Ditto')

    # Set loss function and model. For now, fixed model arch for every task.
    if self.dataset in ('shakespeare'):
      self.data_loss_fn = jax_utils.nll_loss_fn
      self.pred_fn = jax_utils.seq_pred_fn
      self.model_fn, \
        self.mode_prefix, \
          self.last_layers = model_utils.get_model(self.model, 
                                                  self.dataset,
                                                  self.fwd_solver,
                                                  self.bwd_solver)

    else:
      raise ValueError(f'Unsupported dataset: {self.dataset}')

    # Create model architecture & compile prediction/loss function
    self.model  = hk.transform(self.model_fn)
    self.pred_fn = jit(functools.partial(self.pred_fn, self.model))
    self.data_loss_fn = jit(functools.partial(self.data_loss_fn, self.model))

    self.global_iters = 1
    self.local_iters = 1
    step_factor = self.global_iters + self.local_iters

    for t in range(self.num_clients):
      self.batch_gen[t] = gen_batch(self.x_train[t],
                                    self.y_train[t],
                                    self.batch_sizes[t],
                                    num_iter=step_factor * (self.num_rounds + 1) * self.inner_iters)

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:1]
    global_params = self.model.init(key, data_batch)

    num_params = hk.data_structures.tree_size(global_params)
    byte_size = hk.data_structures.tree_bytes(global_params)

    print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')

    local_params = []
    for t in range(self.num_clients):
      local_params.append(self.model.init(key, data_batch))
    local_global_updates = [0] * self.num_clients

    # Optimizer shared for every client (re-init before client work)
    opt = optax.sgd(self.lr)  # provides init_fn, update_fn, params_fn

    def loss_fn(params, batch, rng, global_params):
      mean_batch_term = self.data_loss_fn(params, batch, rng)
      flat_params = jax_utils.model_flatten(params)
      model_diff = flat_params - jax_utils.model_flatten(global_params)
      prox_term = 0.5 * self.lam * (model_diff @ model_diff)
      l2_term = 0.5 * self.l2_reg * (flat_params @ flat_params)
      return mean_batch_term + prox_term + l2_term

    def batch_update(key, params, batch_idx, opt_state, global_params, batch):
      key = random.fold_in(key, batch_idx)
      loss, mean_grad = jax.value_and_grad(loss_fn)(params, 
                                                    batch, 
                                                    key, 
                                                    global_params)

      updates, opt_state = opt.update(mean_grad, opt_state)
      params = optax.apply_updates(params, updates)

      return key, params, opt_state

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
        params = global_params
        opt_state = opt.init(global_params)
        for batch_idx, batch in enumerate(global_batches):
          prox_params = params  # We want prox term to be 0 for global update
          key, params, opt_state = batch_update(key, 
                                                params,
                                                batch_idx, 
                                                opt_state, 
                                                prox_params, 
                                                batch)
          
        new_global_params = params

        # Local updates
        opt_state = opt.init(local_params[t])
        for batch_idx, batch in enumerate(local_batches):
          prox_params = global_params
          key, params, opt_state = batch_update(key,
                                                params,    
                                                batch_idx,
                                                opt_state,
                                                prox_params,
                                                batch)
          
        new_local_params = params

        # Record new *local* model and *global* model diff
        local_global_updates[t] = jax_utils.model_subtract(new_global_params,
                                                           global_params)
        local_params[t] = new_local_params

      # Update global model
      round_local_updates = [local_global_updates[idx] for idx in chosen_idx]
      round_weight_updates = np.asarray([self.update_weights[idx] for idx in chosen_idx])
      average_update = jax_utils.model_average(round_local_updates,
                                               weights=round_weight_updates)
      
      global_params = jax_utils.model_add(global_params, average_update)
      local_global_updates = [0] * self.num_clients

      if i % self.args['eval_every'] == 0:
        train_accu, test_accu = self.eval_seq(local_params, i)

    if self.num_unseen != 0:
      train_accu_unseen, test_accu_unseen = self.eval_unseen(global_params) 

    return train_accu, test_accu

  def eval_seq(self, local_params, round_idx):

    key = random.PRNGKey(self.seed)

    train_losses, test_losses = [], []
    num_correct_train, num_correct_test = [], []
    for t in range(self.num_clients):
      key = random.fold_in(key, t)      
      # Train
      train_acc = self.pred_fn(local_params[t], key, self.x_train[t], self.y_train[t])
      num_correct_train.append(train_acc)
      train_loss = self.data_loss_fn(local_params[t], (self.x_train[t], self.y_train[t]), key)
      train_losses.append(train_loss)
      # Test
      test_acc = self.pred_fn(local_params[t], key, self.x_test[t], self.y_test[t])
      num_correct_test.append(test_acc)
      test_loss  = self.data_loss_fn(local_params[t], (self.x_test[t], self.y_test[t]), key)
      test_losses.append(test_loss)

    avg_train_metric = np.average(num_correct_train, weights=self.train_samples)
    avg_test_metric = np.average(num_correct_test, weights=self.test_samples)
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
  
  def eval_unseen(self, params):

    unseen_params = [params] * self.num_unseen
    key = random.PRNGKey(self.seed)

    unseen_train_losses, unseen_test_losses = [], []
    num_correct_train, num_correct_test = [], []
    for t in range(self.num_unseen):
      key = random.fold_in(key, t)      
      # Train
      train_acc = self.pred_fn(unseen_params[t], key, self.x_unseen_train[t], self.y_unseen_train[t])
      num_correct_train.append(train_acc)
      #train_loss = self.data_loss_fn(unseen_params[t], (self.x_unseen_train[t], self.y_unseen_train[t]), key)
      #unseen_train_losses.append(train_loss)
      # Test
      test_acc = self.pred_fn(unseen_params[t], key, self.x_unseen_test[t], self.y_unseen_test[t])
      num_correct_test.append(test_acc)
      #test_loss  = self.data_loss_fn(unseen_params[t], (self.x_unseen_test[t], self.y_unseen_test[t]), key)
      #unseen_test_losses.append(test_loss)

    avg_unseen_train_metric = np.average(num_correct_train, weights=self.unseen_train_samples)
    avg_unseen_test_metric = np.average(num_correct_test, weights=self.unseen_test_samples)

    #avg_train_loss = np.average(train_losses, weights=self.train_samples)
    #avg_test_loss = np.average(test_losses, weights=self.test_samples)


    if not self.args['quiet']:
      print(f'[Generalization], avg unseen train metric: {avg_unseen_train_metric:.5f},'
            f'avg unseen test metric: {avg_unseen_test_metric:.5f}')

    # Save only 5 decimal places
    data_utils.print_log(np.round([avg_unseen_train_metric, avg_unseen_test_metric], 5).tolist(),
                    stdout=False,
                    fpath=os.path.join(self.args['outdir'], 'unseen.txt'))

    return avg_unseen_train_metric, avg_unseen_test_metric    
  