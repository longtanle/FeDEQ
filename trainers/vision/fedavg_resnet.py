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

from deq.solver import projection

from trainers.base import BaseTrainerGlobal


class FedAvg_Resnet(BaseTrainerGlobal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print(f'[INFO] Running FedAvg with {self.model} model')

    self.model_name = self.model
    # Set loss function and model. For now, fixed model arch for every task.
    if self.dataset in ('femnist', 'cifar10', 'cifar100'):
      self.data_loss_fn = jax_utils.sce_loss_hk_resnet
      self.pred_fn = jax_utils.multiclass_classify_resnet
      self.model_fn, self.mode_prefix, self.last_layers = model_utils.get_model(self.model, 
                                                                                self.dataset,
                                                                                self.fwd_solver,
                                                                                self.bwd_solver)

    else:
      raise ValueError(f'Unsupported dataset: {self.dataset}')

    # Create model architecture & compile prediction/loss function
    self.model  = hk.transform_with_state(self.model_fn)
    self.pred_fn = jit(functools.partial(self.pred_fn, self.model))
    self.data_loss_fn = jit(functools.partial(self.data_loss_fn, self.model))
    self.linf_proj = args['linf_proj']

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    global_params, net_state = self.model.init(key, data_batch, is_training = True)

    num_params = hk.data_structures.tree_size(global_params)
    byte_size = hk.data_structures.tree_bytes(global_params)

    print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')
    #print("trainable:", list(global_params))

    linear_x, shared_params = hk.data_structures.partition(
    lambda m, n, p: m.find("linear_x") != -1, global_params)

    #print("X_params:", list(linear_x))
    linearx_params = hk.data_structures.tree_size(linear_x)
    linearx_byte_size = hk.data_structures.tree_bytes(linear_x)
    #print(f'{linearx_params} params, size: {linearx_byte_size / 1e6:.2f}MB')

    Bz, other_params = hk.data_structures.partition(
    lambda m, n, p: m.find("linear_z") != -1, global_params)

    #print("Z_params:", list(Bz))
    linearz_params = hk.data_structures.tree_size(Bz)
    linearz_byte_size = hk.data_structures.tree_bytes(Bz)
    #print(f'{linearz_params} params, size: {linearz_byte_size / 1e6:.2f}MB')

    # With L_inf projection ball

    def linf_proj_matrix(B, value=1):
        """
        Project a matrix onto an l_inf ball.
        """
        B_proj = jax.vmap(lambda X: projection.projection_l1_sphere(X, value=value), 
                          in_axes=0, 
                          out_axes=0)(B)
        return B_proj

    def projection_linf_params(params, Bname = "linear_z"):

      if self.model_name.find('deq') != -1:  
        Bz, other_params = hk.data_structures.partition(
        lambda m, n, p: m.find(Bname) != -1, params)
        #projected_Bz = projection.projection_linf_ball(Bz, 1.0)

        keys = list(Bz.keys())
        B = Bz[keys[0]]["w"]
        #print("Before: ", B)
        B = linf_proj_matrix(B, value=1)
        #print("After: ",B)
        Bz[keys[0]]["w"] = B

        #projected_Bz = linf_proj_matrix(Bz)
        #print("Infinity norm of B:", tree_inf_norm(Bz))
        #print("Infinity norm of Projected B:", tree_inf_norm(projected_Bz))
        params = hk.data_structures.merge(other_params, Bz)

      return params
    
    local_updates = [0] * self.num_clients
    net_state_updates = [0] * self.num_clients

    if self.linf_proj:
      global_params = projection_linf_params(global_params)

    # Optimizer shared for every client (re-init before client work)
    opt = optax.sgd(self.lr)  # provides init_fn, update_fn, params_fn

    def loss_fn(params, net_state, rng, batch):
      train_term, net_state = self.data_loss_fn(params, net_state, rng, batch)
      l2_term = 0.5 * self.l2_reg * jax_utils.global_l2_norm_sq(params)
      return train_term + l2_term, net_state

    def batch_update(key, params, batch_idx, opt_state, net_state, rng, batch):
      key = random.fold_in(key, batch_idx)
      #params = opt.params_fn(opt_state)
      grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
      (loss, net_state), mean_grad = grad_fn(params, net_state, rng, batch)

      updates, opt_state = opt.update(mean_grad, opt_state)
      params = optax.apply_updates(params, updates)

      if self.linf_proj:
        params = projection_linf_params(params)
      return key, params, opt_state, net_state

    batch_update = jit(batch_update)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[FedAvg_Resnet] Round',
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
          batches = (next(self.batch_gen[t]) for _ in range(self.inner_iters))
        else:
          batches = data_utils.epochs_generator(self.x_train[t],
                                                self.y_train[t],
                                                self.batch_sizes[t],
                                                epochs=self.inner_epochs,
                                                seed=int(key[0]))
        # Local batches
        params = global_params
        opt_state = opt.init(global_params)
        net_state_t = net_state
        for batch_idx, batch in enumerate(batches):
          key, params, opt_state, net_state_t = batch_update(key,
                                                             params,
                                                             batch_idx,
                                                             opt_state,
                                                             net_state_t,
                                                             key,
                                                             batch)
        # Record new model and model diff
        local_updates[t] = jax_utils.model_subtract(params, global_params)
        net_state_updates[t] = jax_utils.model_subtract(net_state_t, net_state)

      # Update global model
      round_local_updates = [local_updates[idx] for idx in chosen_idx]
      round_weight_updates = np.asarray([self.update_weights[idx] for idx in chosen_idx])

      round_net_state_updates = [net_state_updates[idx] for idx in chosen_idx]

      average_update = jax_utils.model_average(round_local_updates,
                                               weights=round_weight_updates)
      average_net_state_update = jax_utils.model_average(round_net_state_updates,
                                                         weights=round_weight_updates)
     
      global_params = jax_utils.model_add(global_params, average_update)
      net_state = jax_utils.model_add(net_state, average_net_state_update)

      local_updates = [0] * self.num_clients
      net_state_updates = [0] * self.num_clients

      if i % self.args['eval_every'] == 0:
        train_accu, test_accu = self.eval_resnet(global_params, net_state, i)


    pickle.dump(global_params, open(os.path.join(self.args['outdir'], "model.pkl"), "wb"))

    if self.num_unseen != 0:
      train_accu_unseen, test_accu_unseen = self.eval_unseen(global_params, net_state)   

    return train_accu, test_accu


  def eval_resnet(self, params, net_state, round_idx):

    local_params = [params] * self.num_clients
    net_states = [net_state] * self.num_clients
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

    avg_unseen_train_metric = np.sum(np.array(num_correct_train)) / np.sum(self.unseen_train_samples)
    avg_unseen_test_metric = np.sum(np.array(num_correct_test)) / np.sum(self.unseen_test_samples)
    
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