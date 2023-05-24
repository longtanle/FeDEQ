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

from deq.solver import projection

from utils.tree_utils import tree_add, tree_sub, tree_scalar_mul, tree_l2_norm, tree_zeros_like, tree_random_normal_like, tree_inf_norm
from trainers.base import BaseTrainerGlobal


class FedAvg(BaseTrainerGlobal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running FedAvg')

    self.model_name = self.model
    # Set loss function and model
    if self.dataset in ('mnist', 'femnist', 'femnist_v2', 'cifar10', 'cifar100'):
      self.data_loss_fn = jax_utils.sce_loss_hk
      self.pred_fn = jax_utils.multiclass_classify
      self.model_fn, \
        self.mode_prefix,\
            self.last_layers = model_utils.get_model(self.model, 
                                                    self.dataset,
                                                    self.fwd_solver,
                                                    self.bwd_solver)

    else:
      raise ValueError(f'Unsupported dataset: {self.dataset}')

    # Create model architecture & compile prediction/loss function
    self.model = hk.without_apply_rng(hk.transform(self.model_fn))
    self.pred_fn = jit(functools.partial(self.pred_fn, self.model))
    self.data_loss_fn = jit(functools.partial(self.data_loss_fn, self.model))
    self.linf_proj = args['linf_proj']

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    global_params = self.model.init(key, data_batch)
    local_updates = [0] * self.num_clients

    print(hk.experimental.tabulate(self.model)(data_batch))

    num_params = hk.data_structures.tree_size(global_params)
    byte_size = hk.data_structures.tree_bytes(global_params)

    print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')
    print("trainable:", list(global_params))

    linear_x, shared_params = hk.data_structures.partition(
    lambda m, n, p: m.find("linear_x") != -1, global_params)

    print("X_params:", list(linear_x))
    linearx_params = hk.data_structures.tree_size(linear_x)
    linearx_byte_size = hk.data_structures.tree_bytes(linear_x)
    print(f'{linearx_params} params, size: {linearx_byte_size / 1e6:.2f}MB')

    Bz, other_params = hk.data_structures.partition(
    lambda m, n, p: m.find("linear_z") != -1, global_params)

    print("Z_params:", list(Bz))
    linearz_params = hk.data_structures.tree_size(Bz)
    linearz_byte_size = hk.data_structures.tree_bytes(Bz)
    print(f'{linearz_params} params, size: {linearz_byte_size / 1e6:.2f}MB')

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

    # projected_Bz = projection.projection_linf_ball(Bz, 1.0)
    if self.linf_proj:
      global_params = projection_linf_params(global_params)

    # Optimizer shared for every client (re-init before client work)
    opt = optax.sgd(self.lr)  # provides init_fn, update_fn, params_fn

    def loss_fn(params, batch):
      train_term = self.data_loss_fn(params, batch)
      l2_term = 0.5 * self.l2_reg * jax_utils.global_l2_norm_sq(params)
      return train_term + l2_term

    def batch_update(key, params, batch_idx, opt_state, batch):
      key = random.fold_in(key, batch_idx)
      #params = opt.params_fn(opt_state)
      mean_grad = grad(loss_fn)(params, batch)
      updates, opt_state = opt.update(mean_grad, opt_state)
      params = optax.apply_updates(params, updates)

      if self.linf_proj:
        params = projection_linf_params(params)

      return key, params, opt_state

    batch_update = jit(batch_update)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[FedAvg] Round',
                  disable=(self.args['repeat'] != 1)):
      key = random.fold_in(key, i)
      chosen_idx = np.random.choice(self.num_clients,
                                    replace=False,
                                    size=self.num_clients_per_round)
      #selected_clients = chosen_idx.tolist()
      selected_clients = list(range(self.num_clients))  # NOTE: use all clients for cross-silo.
      #print(chosen_idx)
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
        local_params = global_params
        opt_state = opt.init(local_params)
        for batch_idx, batch in enumerate(batches):
          key, local_params, opt_state = batch_update(key, local_params, batch_idx, opt_state, batch)
        # Record new model and model diff
        local_updates[t] = jax_utils.model_subtract(local_params, global_params)

      # Update global model
      round_local_updates = [local_updates[idx] for idx in chosen_idx]
      round_weight_updates = np.asarray([self.update_weights[idx] for idx in chosen_idx])
      #print(type(round_local_updates[0]))
      #print(round_weight_updates)
      average_update = jax_utils.model_average(round_local_updates, weights=round_weight_updates)
      global_params = jax_utils.model_add(global_params, average_update)
      #rint(len(global_params))
      local_updates = [0] * self.num_clients

      if i % self.args['eval_every'] == 0:
        train_accu, test_accu = self.eval(global_params, i)

    return train_accu, test_accu
