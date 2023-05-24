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

from trainers.base import BaseTrainerLocal


class LocalResnet(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print(f'[INFO] Running Local with {self.model} model')

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
    
  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    local_params = []
    local_states = []
    for t in range(self.num_clients):
      local_param, local_state = self.model.init(key, data_batch, is_training = True)
      local_params.append(local_param)
      local_states.append(local_state)
      
    # Optimizer shared for every client (re-init before client work)
    # opt = optimizers.momentum(self.lr, mass=0.9)
    opt = optax.sgd(self.lr)  # provides init_fn, update_fn, params_fn

    def loss_fn(params, net_state, rng, batch):
      train_term, net_state = self.data_loss_fn(params, net_state, rng, batch)
      l2_term = 0.5 * self.l2_reg * jax_utils.global_l2_norm_sq(params)
      return train_term + l2_term, net_state

    def batch_update(key, params, batch_idx, opt_state, net_state, rng, batch):
      #params = opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
      (loss, net_state), mean_grad = grad_fn(params, net_state, rng, batch)

      updates, opt_state = opt.update(mean_grad, opt_state)
      params = optax.apply_updates(params, updates)

      return key, params, opt_state, net_state

    batch_update = jit(batch_update)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[Local] Round',
                  disable=(self.args['repeat'] != 1)):  # rounds to run the alternating opt
      key = random.fold_in(key, i)
      selected_clients = list(range(self.num_clients))  # NOTE: use all clients for cross-silo.
      for t in selected_clients:
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
        opt_state = opt.init(local_params[t])
        for batch_idx, batch in enumerate(batches):
          key, local_params[t], \
            opt_state, local_states[t] = batch_update(key,
                                                      local_params[t],
                                                      batch_idx,
                                                      opt_state,
                                                      local_states[t],
                                                      key,
                                                      batch)

        #local_params[t] = opt.params_fn(opt_state)

      if i % self.args['eval_every'] == 0:
        train_accu, test_accu = self.eval_localresnet(local_params, local_states, i)

    return train_accu, test_accu
  
  def eval_localresnet(self, local_params, net_states, round_idx):

    key = random.PRNGKey(self.seed)

    train_losses, test_losses = [], []
    num_correct_train, num_correct_test = [], []
    for t in range(self.num_clients):
      key = random.fold_in(key, t)      
      # Train
      train_preds, _ = self.pred_fn(local_params[t], net_states[t], key, self.x_train[t])
      num_correct_train.append(jnp.sum(train_preds == self.y_train[t]))
      #train_loss, _ = self.data_loss_fn(local_params[t], net_states[t], key, (self.x_train[t], self.y_train[t]))
      #train_losses.append(train_loss)
      # Test
      test_preds, _ = self.pred_fn(local_params[t], net_states[t], key, self.x_test[t])
      num_correct_test.append(jnp.sum(test_preds == self.y_test[t]))
      #test_loss, _  = self.data_loss_fn(local_params[t], net_states[t], key,(self.x_test[t], self.y_test[t]))
      #test_losses.append(test_loss)

    avg_train_metric = np.sum(np.array(num_correct_train)) / np.sum(self.train_samples)
    avg_test_metric = np.sum(np.array(num_correct_test)) / np.sum(self.test_samples)
    #avg_train_loss = np.average(train_losses, weights=self.train_samples)
    #avg_test_loss = np.average(test_losses, weights=self.test_samples)

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