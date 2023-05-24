import os
import functools

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

from trainers.base import BaseTrainerGlobal


class FedRep_SEQ(BaseTrainerGlobal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print(f'[INFO] Running FedRep with {self.model} model')

    # Set loss function and model. For now, fixed model arch for every task.
    if self.dataset in ('shakespeare'):
      self.data_loss_fn = jax_utils.shared_nll_loss_fn
      self.per_data_loss_fn = jax_utils.per_nll_loss_fn
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
    self.per_data_loss_fn = jit(functools.partial(self.per_data_loss_fn, self.model))

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size]
    global_params = self.model.init(key, data_batch)

    # Partition our params into trainable and non trainable explicitly.
    personalized_params, shared_params = hk.data_structures.partition(
    lambda m, n, p: m.find(self.last_layers) != -1, global_params)

    print("Shared_params:", list(shared_params))  
    num_params = hk.data_structures.tree_size(shared_params)
    byte_size = hk.data_structures.tree_bytes(shared_params)
    print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')

    print("Personalized_params:", list(personalized_params))
    num_per_params = hk.data_structures.tree_size(personalized_params)
    byte_size_per = hk.data_structures.tree_bytes(personalized_params)
    print(f'{num_per_params} params, size: {byte_size_per / 1e6:.2f}MB')

    print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')

    local_updates = [0] * self.num_clients

    local_per_params = [personalized_params] * self.num_clients
    local_shared_params = [shared_params] * self.num_clients

    # Optimizer shared for every client (re-init before client work)
    opt = optax.sgd(self.lr) 
    opt_local = optax.sgd(self.lr)

    def shared_loss_fn(shared_params, personalized_params, batch, rng):

      train_term = self.data_loss_fn(shared_params,
                                     personalized_params,
                                     batch,
                                     rng)
      return train_term

    # Define loss functions for personalization
    def personalized_loss_fn(personalized_params,
                             shared_params,
                             rng,
                             batch):
      
      train_term = self.per_data_loss_fn(personalized_params,
                                         shared_params,
                                         rng, batch)
      l2_term = 0.5 * self.l2_reg * jax_utils.global_l2_norm_sq(personalized_params)
      return train_term + l2_term
    
    def batch_update(key, 
                     shared_params,
                     personalized_params,
                     batch_idx, 
                     opt_state, 
                     batch):
      
      key = random.fold_in(key, batch_idx)

      loss, mean_grad = jax.value_and_grad(shared_loss_fn)(shared_params,
                                                           personalized_params,
                                                           batch,
                                                           key)
      
      updates, opt_state = opt.update(mean_grad, opt_state)
      shared_params = optax.apply_updates(shared_params, updates)

      return key, shared_params, opt_state

    batch_update = jit(batch_update)

    def personalized_batch_update(key,
                                  personalized_params,
                                  shared_params,
                                  batch_idx,
                                  opt_state,
                                  batch):

      key = random.fold_in(key, batch_idx)

      grad_local_fn = jax.value_and_grad(personalized_loss_fn)
      loss, mean_local_grad = grad_local_fn(personalized_params,
                                            shared_params,
                                            batch, key)
      updates, opt_state = opt_local.update(mean_local_grad, opt_state)
      personalized_params = optax.apply_updates(personalized_params, updates)

      return key, personalized_params, opt_state
    
    personalized_batch_update = jit(personalized_batch_update)
    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[FedRep] Round',
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
                                           epochs=self.personalized_epochs,
                                           seed=int(key[0]))
          
          shared_batches = data_utils.epochs_generator(self.x_train[t],
                                           self.y_train[t],
                                           self.batch_sizes[t],
                                           epochs=self.inner_epochs,
                                           seed=int(key[0]))
          
        # Server sends current representation to chosen clients 
        local_shared_params[t] = shared_params          
        # Local batches
        opt_state_local = opt_local.init(local_per_params[t])

        for batch_idx, batch in enumerate(batches):
          key, \
            local_per_params[t], \
              opt_state_local = personalized_batch_update(key,
                                                          local_per_params[t],
                                                          local_shared_params[t], 
                                                          batch_idx, 
                                                          opt_state_local,
                                                          batch)
        # Representation Training
        opt_state = opt.init(local_shared_params[t])

        for batch_idx, batch in enumerate(shared_batches):
          key, \
              local_shared_params[t],\
                opt_state = batch_update(key,
                                          local_shared_params[t],
                                          local_per_params[t],
                                          batch_idx,
                                          opt_state,
                                          batch)


      # Server updates the new representation
      round_local_updates = [local_shared_params[idx] for idx in chosen_idx]
      round_weight_updates = np.asarray([self.update_weights[idx] for idx in chosen_idx])
      
      shared_params = jax_utils.model_average(round_local_updates, weights=None)

      if i % self.args['eval_every'] == 0:
        train_accu, test_accu = self.eval_seq(shared_params, local_per_params, i)
    
    if self.num_unseen != 0:
      unseen_shared_params = [shared_params] * self.num_unseen
      unseen_params = [personalized_params] * self.num_unseen

      for t in list(range(self.num_unseen)):  
        key = random.fold_in(key, t)

        # Batch generator
        if self.inner_mode == 'iter':
          unseen_batches = (next(self.batch_gen[t]) for _ in range(self.inner_iters))
        else:
          unseen_batches = data_utils.epochs_generator(self.x_unseen_train[t],
                                                      self.y_unseen_train[t],
                                                      self.unseen_batch_sizes[t],
                                                      epochs=self.personalized_epochs,
                                                      seed=int(key[0]))
        
        # Local batches
        opt_state_local = opt_local.init(unseen_params[t])

        for batch_idx, batch in enumerate(unseen_batches):
          key, unseen_params[t], \
            opt_state_local,  = personalized_batch_update(key, 
                                                          #deq_params,
                                                          unseen_params[t],
                                                          unseen_shared_params[t],
                                                          batch_idx,
                                                          opt_state_local,
                                                          batch)
          
      train_accu_unseen, test_accu_unseen = self.eval_unseen(unseen_shared_params, 
                                                             unseen_params)        

    return train_accu, test_accu


  def eval_seq(self, params, local_per_params, round_idx):

    local_shared_params = [params] * self.num_clients
    key = random.PRNGKey(self.seed)

    train_losses, test_losses = [], []
    num_correct_train, num_correct_test = [], []
    for t in range(self.num_clients):
      key = random.fold_in(key, t)      
      # Train
      t_params = hk.data_structures.merge(local_shared_params[t], local_per_params[t]) 
      train_acc = self.pred_fn(t_params, key, self.x_train[t], self.y_train[t])
      num_correct_train.append(train_acc)
      train_loss = self.per_data_loss_fn(local_per_params[t], local_shared_params[t], (self.x_train[t], self.y_train[t]), key)
      train_losses.append(train_loss)
      # Test
      test_acc = self.pred_fn(t_params, key, self.x_test[t], self.y_test[t])
      num_correct_test.append(test_acc)
      test_loss  = self.per_data_loss_fn(local_per_params[t], local_shared_params[t], (self.x_test[t], self.y_test[t]), key)
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
    """
    if not self.args['is_regression']:
      data_utils.print_log(np.round([avg_train_loss, avg_test_loss], 5).tolist(),
                      stdout=False,
                      fpath=os.path.join(self.args['outdir'], 'losses.txt'))
    """
    return avg_train_metric, avg_test_metric
  

  def eval_unseen(self, unseen_shared_params, unseen_params):

    key = random.PRNGKey(self.seed)

    unseen_train_losses, unseen_test_losses = [], []
    num_correct_train, num_correct_test = [], []
    for t in range(self.num_unseen):
      key = random.fold_in(key, t)      
      # Train
      t_params = hk.data_structures.merge(unseen_shared_params[t], unseen_params[t]) 
      train_acc = self.pred_fn(t_params, key, self.x_unseen_train[t], self.y_unseen_train[t])
      num_correct_train.append(train_acc)
      #train_loss, _ = self.data_loss_fn(t_params, local_net_states[t], key, (self.x_train[t], self.y_train[t]))
      #train_losses.append(train_loss)
      # Test
      test_acc = self.pred_fn(t_params, key, self.x_unseen_test[t], self.y_unseen_test[t])
      num_correct_test.append(test_acc)
      #test_loss, _  = self.data_loss_fn(t_params, local_net_states[t], key,(self.x_test[t], self.y_test[t]))
      #test_losses.append(test_loss)

    #avg_train_metric = np.sum(np.array(num_correct_train)) / np.sum(self.unseen_train_samples)
    #avg_test_metric = np.sum(np.array(num_correct_test)) / np.sum(self.unseen_test_samples)

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
    """
    if not self.args['is_regression']:
      data_utils.print_log(np.round([avg_train_loss, avg_test_loss], 5).tolist(),
                      stdout=False,
                      fpath=os.path.join(self.args['outdir'], 'losses.txt'))
    """

    return avg_unseen_train_metric, avg_unseen_test_metric  