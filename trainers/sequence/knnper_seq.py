import os
import functools
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers
import haiku as hk

import pickle

import optax

from utils import jax_utils
from utils import model_utils
from utils import data_utils

from trainers.base import BaseTrainerGlobal


class kNNPer_SEQ(BaseTrainerGlobal):
  """Implements kNN-Per (https://arxiv.org/pdf/2111.09360.pdf).
  """
  def __init__(self, args, data):
    super().__init__(args, data)
    #print('[INFO] Initializing kNN-Per')
    #self.init_knn_models()
    #self.init_inference_fns()
    print('[INFO] Running kNN-Per')

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

    self.lam_knn = args['knn_lam']

    def _compute_knn_softmax(train_embeddings, train_labels, 
                             eval_embeddings, num_labels):
      """Builds a kNN model, and use it to get a softmax output on the eval set.

      This function builds a *single* kNN model using the embeddings and labels in
      the training set, and evaluate the model on the eval embeddings. For each eval
      example, its prediction is given by the labels of its k nearest neighbors in
      the training set, and the probability is proportional to
      `exp(-d(eval_emb, neighbor_emb))`, where `d` is the Euclidean distance. This
      follows Equation (6) of Marfoq et al., "Personalized Federated Learning
      through Local Memorization", ICML 2022, https://arxiv.org/abs/2111.09360.

      Args:
        train_embeddings: An array of shape (num_train_examples, emb_size).
        train_labels: An array of shape (num_train_examples,).
        eval_embeddings: An array of shape (num_eval_examples, emb_size).
        num_labels: Number of total labels.

      Returns:
        An array of shape (num_eval_examples, num_labels), where each row sums to 1.
      """
      knn_model_fn = KNeighborsRegressor if self.args['is_regression'] else KNeighborsClassifier
      neigh = NearestNeighbors(n_neighbors= self.args['knn_neighbors'])
      neigh.fit(train_embeddings)

      distances, neighbors = neigh.kneighbors(eval_embeddings,
                                              self.args['knn_neighbors'], 
                                              return_distance=True)
      num_eval_examples = eval_embeddings.shape[0]
      knn_softmax = np.zeros((num_eval_examples, num_labels))
      for eval_example_i, (dist_list,
                          nbr_indices) in enumerate(zip(distances, neighbors)):
        nbr_labels = train_labels[nbr_indices]
        for nbr_dist, nbr_label in zip(dist_list, nbr_labels):
          knn_softmax[eval_example_i, nbr_label] += np.exp(-nbr_dist)
        knn_softmax[eval_example_i, :] = knn_softmax[eval_example_i, :] / np.sum(
            knn_softmax[eval_example_i, :])  # Normalize each row to sum to 1.
      return knn_softmax    
    
    def sparse_categorical_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        return jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)    

    def seq_pred_fn(model, params, rng, batch_inputs, batch_labels):
        predictions = model.apply(params, rng, batch_inputs)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == batch_labels)
    
    def softmax_loss(params, rng, batch):
        inputs, targets = batch
        logits, embedding = self.model.apply(params, rng, inputs)

        logits = logits.reshape((-1, 90))
        return jax.nn.softmax(logits) 
  

    if self.dataset in ('mnist', 'femnist', 'cifar10', 'cifar100'):

      def knnper_vision_classify(params, net_state, rng, train_data, test_data):
        """Compute the accuracy of kNN-Per for a single client of a vision dataset.

        Each client's local dataset is split into two sets: a personalization set
        (used to train a kNN model) and an eval set (used to evaluate the personalized
        model). This function works for EMNIST and Landmarks datasets, which use a
        CNN model.

        Args:

        Returns:
          The `SparseCategoricalAccuracy` evaluated on the eval set.
        """
        train_inputs, train_labels = train_data
        test_inputs, test_labels = test_data
        (_, train_embeddings), netstate = self.model.apply(params, net_state, rng, train_inputs)
        (_, eval_embeddings), netstate = self.model.apply(params, net_state, rng, test_inputs, is_training = False)


        global_softmax = softmax_loss(params, net_state, rng, test_data)

        assert np.isclose(np.sum(global_softmax[0, :]), 1.0), (
            'Expected each row of `global_softmax` sums to 1, but the first row sums '
            f'to {np.sum(global_softmax[0, :])}.')
        
        num_labels = global_softmax.shape[-1]
        knn_softmax = _compute_knn_softmax(train_embeddings, train_labels, eval_embeddings, num_labels)        
        personalized_softmax = (self.lam_knn * knn_softmax +  (1.0 - self.lam_knn) * global_softmax)

        accuracy = sparse_categorical_accuracy(test_labels, personalized_softmax)
        return accuracy       

      # NOTE: Cannot JIT because we use scikit kNN models.
      self.knnper_pred_fn = knnper_vision_classify
      # No need to change `data_loss_fn` (hinge loss for Vehicle).
      self.knnper_loss_fn = lambda params, batch, idx: self.data_loss_fn(params, batch)

    elif self.dataset in ('shakespeare'):

      def knn_per_for_language_data(params, rng, train_data, test_data):
        """Compute the accuracy of kNN-Per for a single client of a language dataset.

        Each client's local dataset is split into two sets: a personalization set
        (used to train a kNN model) and an eval set (used to evaluate the personalized
        model). This function works for StackOverflow (use LSTM model) and TedMulti
        (use Transformer model) datasets.
        """
        train_inputs, train_labels = train_data
        test_inputs, test_labels = test_data
        train_logits, train_embeddings = self.model.apply(params, rng, train_inputs)
        eval_logits, eval_embeddings = self.model.apply(params, rng, test_inputs)

        embedding_dim = train_embeddings.shape[-1]
        train_embeddings = train_embeddings.reshape((-1, embedding_dim))
        eval_embeddings = eval_embeddings.reshape((-1, embedding_dim))
        train_labels = train_labels.reshape((-1,))
        test_labels = test_labels.reshape((-1,))

        output_dim = train_logits.shape[-1]

        global_softmax = softmax_loss(params, rng, test_data)

        assert np.isclose(np.sum(global_softmax[0, :]), 1.0), (
            'Expected each row of `global_softmax` sums to 1, but the first row sums '
            f'to {np.sum(global_softmax[0, :])}.')
        
        num_labels = global_softmax.shape[-1]
        knn_softmax = _compute_knn_softmax(train_embeddings, train_labels, eval_embeddings, num_labels)        
        personalized_softmax = (self.lam_knn * knn_softmax +  (1.0 - self.lam_knn) * global_softmax)

        accuracy = sparse_categorical_accuracy(test_labels, personalized_softmax)
        return accuracy     
      # NOTE: Cannot JIT because we use scikit kNN models.
      self.knnper_pred_fn = knn_per_for_language_data
      # No need to change `data_loss_fn` (hinge loss for Vehicle).
      self.knnper_loss_fn = lambda params, batch, idx: self.data_loss_fn(params, batch)            
    else:
      raise ValueError(f'Unsupported dataset: {self.dataset}')

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:1]
    global_params = self.model.init(key, data_batch)

    num_params = hk.data_structures.tree_size(global_params)
    byte_size = hk.data_structures.tree_bytes(global_params)

    print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')
  
    local_updates = [0] * self.num_clients

    # Optimizer shared for every client (re-init before client work)
    opt = optax.sgd(self.lr)
    server_opt = self.server_opt_fn(params_template=global_params)

    def loss_fn(params, batch, rng):
      train_term = self.data_loss_fn(params, batch, rng)
      l2_term = 0.5 * self.l2_reg * jax_utils.global_l2_norm_sq(params)
      return train_term + l2_term

    def batch_update(key, params, batch_idx, opt_state, batch):
      key = random.fold_in(key, batch_idx)
      #params = opt.params_fn(opt_state)
      loss, mean_grad = jax.value_and_grad(loss_fn)(params, batch, key)

      updates, opt_state = opt.update(mean_grad, opt_state)
      params = optax.apply_updates(params, updates)

      return key, params, opt_state

    batch_update = jit(batch_update)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[kNNPer] Round',
                  disable=(self.args['repeat'] != 1)):
      key = random.fold_in(key, i)

      chosen_idx = np.random.choice(self.num_clients,
                                    replace=False,
                                    size=self.num_clients_per_round)

      for t in chosen_idx:
        key = random.fold_in(key, t)
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
        opt_state = opt.init(params)

        for batch_idx, batch in enumerate(batches):
          #print('Batch IDX ', batch_idx)
          key, params, opt_state = batch_update(key,
                                                params,
                                                batch_idx,
                                                opt_state,
                                                batch)

        # Record new model and model diff
        local_updates[t] = jax_utils.model_subtract(params, global_params)

      # Update global model and reset local updates
      # Update global model
      round_local_updates = [local_updates[idx] for idx in chosen_idx]
      round_weight_updates = np.asarray([self.update_weights[idx] for idx in chosen_idx])

      average_update = jax_utils.model_average(round_local_updates,
                                               weights=round_weight_updates)
     

      global_params = jax_utils.model_add(global_params, average_update)

      local_updates = [0] * self.num_clients

      # NOTE: since eval does not JIT (we rely on scikit-learn kNN impl), it is
      # a performance bottleneck.
      if i % self.args['eval_every'] == 0:
        client_metrics = self.knn_per_avg_clients(global_params, i)  # (K, 3)

    pickle.dump(global_params, open(os.path.join(self.args['outdir'], "model.pkl"), "wb"))

    if self.num_unseen != 0:
      unseen_client_metrics = self.eval_unseen(global_params) 
    # Return metric at final round
    return client_metrics

  ##############################################################################


  def init_knn_models(self):
    """Create kNN models from each client's training set."""
    knn_model_fn = KNeighborsRegressor if self.args['is_regression'] else KNeighborsClassifier
    self.knn_models = []
    for t in range(self.num_clients):
      model = knn_model_fn(n_neighbors=self.args['knn_neighbors'],
                           weights=self.args['knn_weights'])
      model = model.fit(self.x_train[t], self.y_train[t])
      self.knn_models.append(model)


  def init_inference_fns(self):
    """(Re-)initializes the inference functions (loss and prediction) for kNN-Per."""

  @jit  
  def vision_model_pred(self, params, net_state, rng, batch_inputs):
    logits, net_state = self.model.apply(params, net_state, rng, batch_inputs, is_training = False)
    return logits.squeeze()

  @jit
  def language_model_pred(self, params, rng, batch_inputs):
    predictions = self.model.apply(params, rng, batch_inputs)
    return predictions.squeeze()
  

  def knn_per_avg_clients(self, params, round_idx):
    """Compute the per-client accuracy of kNN-Per for a list of clients."""
    local_params = [params] * self.num_clients  
    key = random.PRNGKey(self.seed)

    client_accs = []
    avg_train_accs = []
    avg_test_accs = []
    for t in range(self.num_clients):
      key = random.fold_in(key, t)  
      """
      train_acc = self.knnper_pred_fn(train_data=(self.x_train[t], self.y_train[t]),
                                     test_data=(self.x_train[t], self.y_train[t]),
                                     params=local_params[t],
                                     net_state = local_net_states[t],
                                     rng = key)
      """                               
            
      test_acc = self.knnper_pred_fn(train_data=(self.x_train[t], self.y_train[t]),
                                     test_data=(self.x_test[t], self.y_test[t]),
                                     params=local_params[t],
                                     #net_state = local_net_states[t],
                                     rng = key)
      
      #avg_train_accs.append(np.mean(train_acc))
      avg_test_accs.append(np.mean(test_acc))

      client_accs.append(test_acc)

    #avg_train_metric =  np.mean(avg_train_accs)
    avg_test_metric =  np.mean(avg_test_accs)

    if not self.args['quiet']:
      print(f'Round {round_idx}, avg test metric: {avg_test_metric:.5f}')


    # Save only 5 decimal places
    data_utils.print_log(np.round([avg_test_metric], 5).tolist(),
                    stdout=False,
                    fpath=os.path.join(self.args['outdir'], 'output.txt'))
      
    return avg_test_metric

  def eval_unseen(self, params):

    unseen_params = [params] * self.num_unseen 
    key = random.PRNGKey(self.seed)

    unseen_client_accs = []
    avg_train_accs = []
    avg_unseen_test_accs = []
    for t in range(self.num_unseen):
      key = random.fold_in(key, t)  
      """
      train_acc = self.knnper_pred_fn(train_data=(self.x_train[t], self.y_train[t]),
                                    test_data=(self.x_train[t], self.y_train[t]),
                                    params=local_params[t],
                                    net_state = local_net_states[t],
                                    rng = key)
      """                               
            
      unseen_test_acc = self.knnper_pred_fn(train_data=(self.x_unseen_train[t], self.y_unseen_train[t]),
                                            test_data=(self.x_unseen_test[t], self.y_unseen_test[t]),
                                            params=unseen_params[t],
                                            #net_state = unseen_net_states[t],
                                            rng = key)
      
      #avg_train_accs.append(np.mean(train_acc))
      avg_unseen_test_accs.append(np.mean(unseen_test_acc))


      unseen_client_accs.append(unseen_test_acc)

    #avg_train_metric =  np.mean(avg_train_accs)
    avg_unseen_test_metric =  np.mean(avg_unseen_test_accs)

    if not self.args['quiet']:
      print(f'[Generalization], avg unseen test metric: {avg_unseen_test_metric:.5f}') 

    # Save only 5 decimal places
    data_utils.print_log(np.round([avg_unseen_test_metric], 5).tolist(),
                    stdout=False,
                    fpath=os.path.join(self.args['outdir'], 'unseen.txt'))
    
    return avg_unseen_test_metric
  
  def eval_knnper(self,
                  global_params,
                  round_idx,
                  save=True,
                  save_per_client=True,
                  fn_prefix='',
                  quiet=False):
    """HACK: Custom evaluation function for kNN-Per, as inference uses local datastores.

    The only change is on the calls to `data_loss_fn` and `pred_fn`, which now
    includes the client index for accessing the local kNN models.
    """
    local_params = [global_params] * self.num_clients
    # Compute loss (both regression and classification)
    quiet = quiet or self.args['quiet']
    if self.args['no_per_client_metric']:
      save_per_client = False

    outdir = Path(self.args['outdir'])
    losses = []
    for t in range(self.num_clients):
      # HACK: allow loss_fn to take client index.
      train_loss = self.knnper_loss_fn(local_params[t], (self.x_train[t], self.y_train[t]), t)
      val_loss = self.knnper_loss_fn(local_params[t], (self.x_val[t], self.y_val[t]), t)
      test_loss = self.knnper_loss_fn(local_params[t], (self.x_test[t], self.y_test[t]), t)
      losses.append((train_loss, val_loss, test_loss))

    losses = np.array(losses).astype(float)  # (K, 3); float64 for rounding str
    # Unweighted averaging of client metrics
    avg_loss, std_loss = np.mean(losses, axis=0), np.std(losses, axis=0)  # (3,)

    if self.args['is_regression']:
      metrics, avg_metric, std_metric = losses, avg_loss, std_loss
    else:
      # Classification additionally computes accuracy
      accs = []
      for t in range(self.num_clients):
        # HACK: allow pred_fn to take client index.
        train_acc = np.mean(self.knnper_pred_fn(local_params[t], self.x_train[t], t) == self.y_train[t])
        val_acc = np.mean(self.knnper_pred_fn(local_params[t], self.x_val[t], t) == self.y_val[t])
        test_acc = np.mean(self.knnper_pred_fn(local_params[t], self.x_test[t], t) == self.y_test[t])
        accs.append((train_acc, val_acc, test_acc))

      metrics = accs = np.array(accs).astype(float)  # (K, 3)
      avg_metric = avg_acc = np.mean(accs, axis=0)  # (3,)
      std_metric = std_acc = np.std(accs, axis=0)  # (3,)
      # Also save loss when doing classification
      if save:
        if save_per_client:
          data_utils.print_log(np.round(losses, 5).tolist(),
                          fpath=outdir / f'{fn_prefix}client_losses.txt')
        data_utils.print_log(np.round(avg_loss, 5).tolist(), fpath=outdir / f'{fn_prefix}avg_losses.txt')
        data_utils.print_log(np.round(std_loss, 5).tolist(), fpath=outdir / f'{fn_prefix}std_losses.txt')

    # Save avg / per-client metrics (both classification and regression)
    assert metrics.shape == (self.num_clients, 3)
    if save:
      if save_per_client:
        data_utils.print_log(np.round(metrics, 5).tolist(),
                        fpath=outdir / f'{fn_prefix}client_metrics.txt')
      data_utils.print_log(np.round(avg_metric, 5).tolist(),
                      fpath=outdir / f'{fn_prefix}avg_metrics.txt')
      data_utils.print_log(np.round(std_metric, 5).tolist(),
                      fpath=outdir / f'{fn_prefix}std_metrics.txt')

    if not quiet:
      print(f'Round {round_idx}, avg metric train/val/test: {np.round(avg_metric, 5)}')

    return metrics  # (K, 3)
