import functools
import os

import numpy as np
from tqdm import tqdm

from jax import jit
import jax.numpy as jnp
import haiku as hk
from jax import random, jit, grad, vmap

from utils.data_utils import gen_batch, print_log
from utils import jax_utils
from utils import model_utils
from utils import data_utils
from utils import opt_utils

class BaseTrainer:
  def __init__(self, args, data):
    x_train, y_train, x_val, y_val, x_test, y_test = data
    self.args = args
    self.seed = args['seed']

    ###### Client #######
    self.data = data
    self.num_total_clients = len(x_train)  # number of clients / tasks
    self.frac_unseen = args['frac_unseen']
    self.num_unseen =  int(self.frac_unseen * self.num_total_clients)
    self.num_clients = self.num_total_clients - self.num_unseen 
    self.clients_per_round = args['clients_per_round']
    self.num_clients_per_round = int((self.clients_per_round * self.num_clients) // 1)


    print('[INFO] number of training clients per round', self.num_clients_per_round)

    if self.num_unseen != 0:
      print('[INFO] number of unseen clients', self.num_unseen)

      chosen_idx = np.random.choice(self.num_clients, 
                              replace=False,
                              size=self.num_unseen)
      
      self.x_train  = x_train[:self.num_clients]
      self.y_train  = y_train[:self.num_clients]

      self.x_unseen_train = x_train[self.num_clients:self.num_total_clients]
      self.y_unseen_train = y_train[self.num_clients:self.num_total_clients]
      
      #self.unseen_train_samples = self.train_samples[self.num_clients:self.num_total_clients]
      #self.train_samples  = self.train_samples[:self.num_clients]
      self.x_test  = x_test[:self.num_clients]
      self.y_test  = y_test[:self.num_clients]

      self.x_unseen_test = x_test[self.num_clients:self.num_total_clients]
      self.y_unseen_test = y_test[self.num_clients:self.num_total_clients]

      self.x_val, self.y_val = x_val, y_val
    else:
      self.x_unseen_train = self.y_unseen_train = self.x_unseen_test = self.y_unseen_test = []
      self.x_train, self.y_train = x_train, y_train
      self.x_test, self.y_test = x_test, y_test
      self.x_val, self.y_val = x_val, y_val

    ###### Training configs #######
    self.model = args['model']
    self.lam = args['lambda']
    self.num_rounds = args['num_rounds']
    self.inner_mode = args['inner_mode']
    self.inner_epochs = args['inner_epochs']
    self.inner_iters = args['inner_iters']
    self.personalized_epochs = args['personalized_epochs']
    self.lr = args['learning_rate']
    self.l2_reg = args['l2_reg']
    self.dataset = args['dataset']
    self.gpu = args['gpu']
    self.train_samples = np.asarray([len(x) for x in self.x_train])
    self.test_samples = np.asarray([len(x) for x in self.x_test])
    self.val_samples = np.asarray([len(x) for x in self.x_val])
    self.unseen_train_samples = np.asarray([len(x) for x in self.x_unseen_train])
    self.unseen_test_samples = np.asarray([len(x) for x in self.x_unseen_test])
    #self.l2_clip = self.args['ex_clip']
    #self.num_clusters = args['num_clusters']

    if args['unweighted_updates']:
      self.update_weights = np.ones_like(self.train_samples)
    else:
      self.update_weights = self.train_samples / np.sum(self.train_samples)

    self.batch_size = args['batch_size']
    if self.batch_size == -1:
      # Full batch gradient descent if needed
      self.batch_sizes = [len(self.x_train[i]) for i in range(self.num_clients)]
      self.unseen_batch_sizes = [len(self.x_unseen_train[i]) for i in range(self.num_unseen)]
    else:
      # Limit batch size to the dataset size, so downstream calculations (e.g. sample rate) don't break
      self.batch_sizes = [min(len(self.x_train[i]), self.batch_size) for i in range(self.num_clients)]
      self.unseen_batch_sizes = [min(len(self.x_unseen_train[i]), self.batch_size) for i in range(self.num_unseen)]

    self.per_batch_size = args['per_batch_size']
    if self.per_batch_size == -1:
      # Full batch gradient descent if needed
      self.per_batch_sizes = [len(self.x_train[i]) for i in range(self.num_clients)]

    else:
      # Limit batch size to the dataset size, so downstream calculations (e.g. sample rate) don't break
      self.per_batch_sizes = [min(len(self.x_train[i]), self.per_batch_size) for i in range(self.num_clients)]


    ###### Server Optimizer ######
    if not self.args['quiet']:
      print(f'[INFO] Server optimizer: {self.args["server_opt"]}')
    if self.args['server_opt'] == 'fedavgm':
      self.server_opt_fn = functools.partial(opt_utils.FedAvgM,
                                             server_lr=self.args['server_lr'],
                                             momentum=self.args['fedavg_momentum'])
    elif self.args['server_opt'] == 'fedadam':
      self.server_opt_fn = functools.partial(opt_utils.FedAdam,
                                             server_lr=self.args['server_lr'],
                                             beta_1=self.args['fedadam_beta1'],
                                             beta_2=self.args['fedadam_beta2'],
                                             tau=self.args['fedadam_tau'])
    else:  # Fallback to FedAvg
      self.server_opt_fn = functools.partial(opt_utils.FedAvg, server_lr=self.args['server_lr'])


    ###### DP configs #######
    #self.use_dp = self.args['example_dp']
    # NOTE: hack: trying to save compute as private selection needs to recompute noise_mults.
    #if 'ifca' not in self.args['trainer'].lower():
    #  self.noise_mults = dp_accounting.compute_silo_noise_mults(self.num_clients, self.train_samples, args)
    
    ###### DEQ configs #######
    self.fwd_solver = args['fwd_solver']
    self.bwd_solver = args['bwd_solver']

    # (DEPRECATED) Batch-wise local data generators; deprecated to use local epochs.
    self.batch_gen = {}
    for i in range(self.num_clients):
      self.batch_gen[i] = gen_batch(self.x_train[i], self.y_train[i], self.batch_size,
                                    num_iter=(self.num_rounds + 1) * self.inner_iters)
    self.train_data = self.batch_gen  # Legacy.

    with np.printoptions(precision=4):
      print('[DEBUG] client update weights', self.update_weights)

  def train(self):
    raise NotImplementedError(f'BaseTrainer train() needs to be implemented')

  def eval(self, local_params, round_idx):
    # Regression
    if self.args['is_regression']:
      train_losses, test_losses = [], []
      for t in range(self.num_clients):
        train_preds = self.pred_fn(local_params[t], self.x_train[t])
        train_losses.append(np.mean((train_preds - self.y_train[t])**2))
        test_preds = self.pred_fn(local_params[t], self.x_test[t])
        test_losses.append(np.mean((test_preds - self.y_test[t])**2))
      avg_train_metric = np.average(train_losses, weights=self.train_samples)
      avg_test_metric = np.average(test_losses, weights=self.test_samples)

    # Classification
    else:
      train_losses, test_losses = [], []
      num_correct_train, num_correct_test = [], []
      for t in range(self.num_clients):
        # Train
        train_preds = self.pred_fn(local_params[t], self.x_train[t])
        num_correct_train.append(jnp.sum(train_preds == self.y_train[t]))
        train_loss = self.data_loss_fn(local_params[t], (self.x_train[t], self.y_train[t]))
        train_losses.append(train_loss)
        # Test
        test_preds = self.pred_fn(local_params[t], self.x_test[t])
        num_correct_test.append(jnp.sum(test_preds == self.y_test[t]))
        test_loss = self.data_loss_fn(local_params[t], (self.x_test[t], self.y_test[t]))
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
   

class BaseTrainerLocal(BaseTrainer):
  def __init__(self, args, data):
    super().__init__(args, data)


class BaseTrainerGlobal(BaseTrainer):
  def __init__(self, params, data):
    super().__init__(params, data)

  def eval(self, params, round_idx):
    local_params = [params] * self.num_clients
    return super().eval(local_params, round_idx)

