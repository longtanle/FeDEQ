import typing as tp

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, device_put
import optax
import haiku as hk
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax.scipy.special import expit as sigmoid
import numpy as np

from jaxopt import loss
from jaxopt import projection
from jaxopt.tree_util import tree_add, tree_sub, tree_vdot, tree_dot, tree_l2_norm
from utils import jax_utils

logistic_loss = jax.vmap(loss.multiclass_logistic_loss)

def predict(model, all_params, state, rng, images, train=False):
  """Forward pass in the network on the images."""
  #x = images.astype(jnp.float32) / 255.
  #mutable = ["batch_stats"] if train else False
  return model.apply(all_params, state, rng, images, train)

def loss_from_logits(params, l2reg, logits, labels):
  sqnorm = tree_l2_norm(params, squared=True)
  mean_loss = jnp.mean(logistic_loss(labels, logits))
  return mean_loss + 0.5 * l2reg * sqnorm

def accuracy_and_loss(model, params, model_state, rng, l2reg, data):
  #all_vars = {"params": params, "batch_stats": aux}
  images, labels = data
  logits, net_state = predict(model, params, model_state, rng, images)
  accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
  loss = loss_from_logits(params, l2reg, logits, labels)
  return accuracy, loss

############################################
############## Training utils ##############
############################################

def lm_loss_hk(model,
               params,
               rng,
               vocab_size: int,
               data: tp.Mapping[str, jnp.ndarray],
               is_training: bool = True) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    logits = model.apply(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['target'], vocab_size)
    assert logits.shape == targets.shape

    mask = jnp.greater(data['obs'], 0)
    log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)

    return -jnp.sum(log_likelihood * mask) / jnp.sum(mask)  # NLL per token.


def hinge_loss_hk(model, params, batch, reg=0.1):
  inputs, targets = batch  # (n, ...), (n,)
  param_vec = model_flatten(params)
  # Scalar output's last dimension needs to be squeezed.
  preds = model.apply(params=params, inputs=inputs).squeeze()
  losses = jax.nn.relu(1.0 - targets * preds) + 0.5 * reg * (param_vec @ param_vec)
  return jnp.mean(losses)


def bce_loss_hk(model, params, batch):
  inputs, targets = batch
  # Scalar output's last dimension needs to be squeezed.
  logits = model.apply(params=params, inputs=inputs).squeeze()
  per_example_loss = optax.sigmoid_binary_cross_entropy(logits, targets)
  return jnp.mean(per_example_loss)

#########################
#### For Vision Tasks ###
#########################

def sce_loss_hk(model, params, batch):
  inputs, targets = batch
  (logits, embedding) = model.apply(params=params, inputs=inputs)
  targets = jax.nn.one_hot(targets, logits.shape[-1])  # Conver to one_hot
  per_example_loss = optax.softmax_cross_entropy(logits, targets)
  return jnp.mean(per_example_loss)  # `mean` implicitly means away the batch dim too.

def sce_loss_hk_resnet(model, params, net_state, rng, batch):
  inputs, targets = batch
  (logits, embedding), net_state = model.apply(params, net_state, rng, inputs, is_training = True)
  targets = jax.nn.one_hot(targets, logits.shape[-1])  # Conver to one_hot
  per_example_loss = optax.softmax_cross_entropy(logits, targets)
  return jnp.mean(per_example_loss), net_state  # `mean` implicitly means away the batch dim too.

def shared_loss_hk_resnet(model, shared_params, personalized_params, net_state, rng, batch):
  inputs, targets = batch
  params = hk.data_structures.merge(shared_params, personalized_params) 
  (logits, embedding), net_state = model.apply(params, net_state, rng, inputs, is_training = True)
  targets = jax.nn.one_hot(targets, logits.shape[-1])  # Conver to one_hot

  per_example_loss = optax.softmax_cross_entropy(logits, targets) 
  return jnp.mean(per_example_loss), net_state  # `mean` implicitly means away the batch dim too.

def per_sce_loss_hk_resnet(model, personalized_params, deq_params, net_state, rng, batch):
  inputs, targets = batch
  params = hk.data_structures.merge(deq_params, personalized_params) 
  (logits, embedding), net_state = model.apply(params, net_state, rng, inputs, is_training = True)
  targets = jax.nn.one_hot(targets, logits.shape[-1])  # Conver to one_hot
  per_example_loss = optax.softmax_cross_entropy(logits, targets)
  return jnp.mean(per_example_loss), net_state  # `mean` implicitly means away the batch dim too.

########################
#### For FEDEQ Tasks ###
########################
def perdeq_sce_loss_hk_resnet(model, personalized_params, deq_params, net_state, rng, batch):
  inputs, targets = batch
  params = hk.data_structures.merge(deq_params, personalized_params) 
  (logits, embedding), net_state = model.apply(params, net_state, rng, inputs, is_training = True)
  targets = jax.nn.one_hot(targets, logits.shape[-1])  # Conver to one_hot
  per_example_loss = optax.softmax_cross_entropy(logits, targets)
  return jnp.mean(per_example_loss), net_state  # `mean` implicitly means away the batch dim too.

def surrogate_loss_hk_resnet(model, deq_params, personalized_params, shared_params, net_state, rng, batch, lam, rho):
  inputs, targets = batch
  params = hk.data_structures.merge(deq_params, personalized_params) 
  (logits, embedding), net_state = model.apply(params, net_state, rng, inputs, is_training = True)
  targets = jax.nn.one_hot(targets, logits.shape[-1])  # Conver to one_hot

  linear_term = tree_vdot(lam, jax_utils.model_subtract(deq_params, shared_params))
  quadratic_term = 0.5 * rho * tree_l2_norm(jax_utils.model_subtract(deq_params, shared_params), squared=True)
  
  per_example_loss = optax.softmax_cross_entropy(logits, targets) 
  return jnp.mean(per_example_loss) + linear_term + quadratic_term, net_state  # `mean` implicitly means away the batch dim too.

def smooth_per_loss_hk_deq(model, personalized_params, deq_params, net_state, rng, batch, smoothing=0.1):
  inputs, targets = batch
  params = hk.data_structures.merge(deq_params, personalized_params) 
  (logits, embedding), net_state = model.apply(params, net_state, rng, inputs, is_training = True)
  #targets = jax.nn.one_hot(targets, logits.shape[-1])  # Conver to one_hot
  num_classes = logits.shape[-1]
  # Apply label smoothing to the target labels
  confidence = 1.0 - smoothing
  low_confidence = smoothing / (num_classes - 1)
  smoothed_labels = jnp.full_like(logits, low_confidence).at[:, targets].add(confidence - low_confidence)

  per_example_loss = optax.softmax_cross_entropy(logits, smoothed_labels)
  return jnp.mean(per_example_loss), net_state  # `mean` implicitly means away the batch dim too.

def smooth_shared_loss_hk_deq(model, deq_params, personalized_params, shared_params, net_state, rng, batch, lam, rho, smoothing=0.1):
  inputs, targets = batch
  params = hk.data_structures.merge(deq_params, personalized_params) 
  (logits, embedding), net_state = model.apply(params, net_state, rng, inputs, is_training = True)
  #targets = jax.nn.one_hot(targets, logits.shape[-1])  # Conver to one_hot
  num_classes = logits.shape[-1]
  # Apply label smoothing to the target labels
  confidence = 1.0 - smoothing
  low_confidence = smoothing / (num_classes - 1)
  smoothed_labels = jnp.full_like(logits, low_confidence).at[:, targets].add(confidence - low_confidence)


  linear_term = tree_vdot(lam, jax_utils.model_subtract(deq_params, shared_params))
  quadratic_term = 0.5 * rho * tree_l2_norm(jax_utils.model_subtract(deq_params, shared_params), squared=True)
  
  per_example_loss = optax.softmax_cross_entropy(logits, smoothed_labels) 
  return jnp.mean(per_example_loss) - linear_term + quadratic_term, net_state  # `mean` implicitly means away the batch dim too.


def sce_loss_hk_deq(model, params, state, rng, l2reg, batch):
  inputs, targets = batch
  # Predict
  logits, net_state = model.apply(params, state, rng, inputs)
  #accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
  loss = loss_from_logits(params, l2reg, logits, targets)
  #targets = jax.nn.one_hot(targets, logits.shape[-1])  # Conver to one_hot
  #per_example_loss = optax.softmax_cross_entropy(logits, targets)
  return loss, net_state  # `mean` implicitly means away the batch dim too.

########################
## For Sequence Tasks ##
########################
def nll_loss_fn(model, params, batch, rng, vocab_size = 90):
  """Computes the (scalar) LM loss on `data` w.r.t. params."""
  inputs, targets = batch

  logits, embedding = model.apply(params, rng, inputs)
  targets = jax.nn.one_hot(targets, vocab_size)
  assert logits.shape == targets.shape

  mask = jnp.greater(inputs, 0)
  log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
  return -jnp.sum(log_likelihood * mask) / jnp.sum(mask)  # NLL per token.

def shared_nll_loss_fn(model, shared_params, personalized_params, batch, rng, vocab_size = 90):
  """Computes the (scalar) LM loss on `data` w.r.t. params."""
  inputs, targets = batch

  params = hk.data_structures.merge(shared_params, personalized_params)
  logits, embedding = model.apply(params, rng, inputs)
  targets = jax.nn.one_hot(targets, vocab_size)
  assert logits.shape == targets.shape

  mask = jnp.greater(inputs, 0)
  log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
  return -jnp.sum(log_likelihood * mask) / jnp.sum(mask)  # NLL per token.

def rdeq_nll_loss_fn(model,
                     deq_params, 
                     personalized_params,
                     shared_params,
                     batch, rng, 
                     lam, rho,
                     vocab_size = 90):
  """Computes the (scalar) LM loss on `data` w.r.t. params."""
  inputs, targets = batch

  params = hk.data_structures.merge(deq_params, personalized_params)
  logits, embedding = model.apply(params, rng, inputs)
  targets = jax.nn.one_hot(targets, vocab_size)
  assert logits.shape == targets.shape

  diff_params = jax_utils.model_subtract(deq_params, shared_params)

  linear_term = tree_vdot(lam, diff_params)
  quadratic_term = 0.5 * rho * tree_l2_norm(diff_params, squared=True)

  mask = jnp.greater(inputs, 0)
  #print(jnp.sum(mask))
  log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
  return -jnp.sum(log_likelihood * mask)/jnp.sum(mask) + linear_term/jnp.sum(mask)  + quadratic_term/jnp.sum(mask)  # NLL per token.


def perdeq_nll_loss_fn(model, personalized_params, shared_params, batch, rng, vocab_size = 90):
  """Computes the (scalar) LM loss on `data` w.r.t. params."""
  inputs, targets = batch

  params = hk.data_structures.merge(shared_params, personalized_params)

  logits, embedding = model.apply(params, rng, inputs)
  targets = jax.nn.one_hot(targets, vocab_size)
  assert logits.shape == targets.shape

  mask = jnp.greater(inputs, 0)
  log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
  return -jnp.sum(log_likelihood * mask) / jnp.sum(mask)  # NLL per token.
  

def per_nll_loss_fn(model, personalized_params, shared_params, batch, rng, vocab_size = 90):
  """Computes the (scalar) LM loss on `data` w.r.t. params."""
  inputs, targets = batch

  params = hk.data_structures.merge(shared_params, personalized_params)

  logits, embedding = model.apply(params, rng, inputs)
  targets = jax.nn.one_hot(targets, vocab_size)
  assert logits.shape == targets.shape

  mask = jnp.greater(inputs, 0)
  log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
  return -jnp.sum(log_likelihood * mask) / jnp.sum(mask)  # NLL per token.
  
def seq_perplexity(model, params, rng, batch_inputs, batch_labels, vocab_size = 90):
   
  predictions = model.apply(params, rng, batch_inputs)
  targets = jax.nn.one_hot(batch_labels, vocab_size)
  assert predictions.shape == targets.shape
  
  total_log_perplexity = jnp.sum(predictions * targets, axis= -1)
  equals_zero = jnp.equal(batch_labels, 0)
  non_pad = 1.0 - equals_zero
  real_log_perplexity = total_log_perplexity * non_pad
  return -jnp.sum(real_log_perplexity) / jnp.sum(non_pad)

########################
#### For Other Tasks ###
########################

def l2_loss_hk(model, params, batch):
  inputs, targets = batch
  # Scalar output's last dimension needs to be squeezed.
  preds = model.apply(params=params, inputs=inputs).squeeze()
  per_example_loss = 0.5 * (preds - targets)**2
  return jnp.mean(per_example_loss)


def global_l2_norm_sq(tensor_struct):
  # NOTE: Apparently you can get NaNs from `jnp.linalg.norm`; the gist is
  # that `norm` is not differentiable at 0, but `squared-norm` is indeed
  # differentiable at 0 (needed for l2 regularization).
  # https://github.com/google/jax/issues/3058
  # https://github.com/google/jax/issues/6484
  flat_vec = model_flatten(tensor_struct)
  return flat_vec @ flat_vec


def global_l2_clip(tensor_struct, clip: float):
  t_list, tree_def = tree_flatten(tensor_struct)
  global_norm = jnp.linalg.norm([jnp.linalg.norm(t.reshape(-1), ord=2) for t in t_list])
  norm_factor = jnp.minimum(clip / (global_norm + 1e-15), 1.0)
  clipped_t_list = [t * norm_factor for t in t_list]
  return tree_unflatten(tree_def, clipped_t_list)


def privatize_grad_hk(example_grads, key, clip, noise_mult):
  # Clipping
  clip_fn = vmap(global_l2_clip, in_axes=(0, None), out_axes=0)
  example_grads = clip_fn(example_grads, clip)
  # Sum
  flat_example_grads, tree_def = tree_flatten(example_grads)
  batch_size = len(flat_example_grads[0])  # 1st dim of per-example grad tensors
  flat_sum_grads = [g.sum(axis=0) for g in flat_example_grads]
  # Noise & mean
  keys = random.split(key, len(flat_sum_grads))
  flat_mean_noisy_grads = [(g + clip * noise_mult * random.normal(k, g.shape)) / batch_size
                           for k, g in zip(keys, flat_sum_grads)]
  return tree_unflatten(tree_def, flat_mean_noisy_grads)

########################
#### Prediction Fn #####
########################

def multiclass_classify(model, params, batch_inputs):
  (logits, embedding) = model.apply(params=params, inputs=batch_inputs)
  pred_class = jnp.argmax(logits, axis=1)
  return pred_class

def multiclass_classify_resnet(model, params, net_state, rng, batch_inputs):
  (logits, embedding), net_state = model.apply(params,
                                               net_state,
                                               rng,
                                               batch_inputs,
                                               is_training = False)
  pred_class = jnp.argmax(logits, axis=1)
  return pred_class, net_state
  
def multiclass_classify_deq(model, params, net_state, rng, batch_inputs):
  (logits, embedding), net_state = model.apply(params,
                                               net_state,
                                               rng,
                                               batch_inputs,
                                               is_training = False)
  pred_class = jnp.argmax(logits, axis=1)
  return pred_class, net_state


def linear_svm_classify(model, params, batch_inputs):
  preds = model.apply(params=params, inputs=batch_inputs).squeeze()
  return jnp.sign(preds)


def logreg_classify(model, params, batch_inputs, temperature=1.0):
  # data_x: x: (n, d), w: (d,), b: (1) --> out: (n,)
  preds = model.apply(params=params, inputs=batch_inputs).squeeze()
  preds = sigmoid(preds / temperature)
  return jnp.round(preds)

def regression_pred(model, params, batch_inputs):
  return model.apply(params=params, inputs=batch_inputs).squeeze()

def lstm_pred_fn(model, params, batch_inputs):
  logits = model.apply(params, batch_inputs)
  pred_class = jnp.argmax(logits, axis=1)
  return pred_class

def seq_pred_fn(model, params, rng, batch_inputs, batch_labels):
    predictions, embedding = model.apply(params, rng, batch_inputs)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch_labels)

def deq_pred_fn(model, params, rng, batch_inputs, batch_labels):
    predictions, embedding = model.apply(params, rng, batch_inputs)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch_labels)

##########################################
############## Struct utils ##############
##########################################


def num_params(tensor_struct):
  param_list, _ = tree_flatten(tensor_struct)
  return np.sum([w.size for w in param_list])  # Use numpy since shape is static.


@jit
def model_add(params_1, params_2):
  return tree_map(jnp.add, params_1, params_2)


@jit
def model_subtract(params_1, params_2):
  return tree_map(jnp.subtract, params_1, params_2)


@jit
def model_multiply(params_1, params_2):
  return tree_map(jnp.multiply, params_1, params_2)


@jit
def model_sqrt(params):
  return tree_map(jnp.sqrt, params)


@jit
def model_divide(params_1, params_2):
  return tree_map(jnp.divide, params_1, params_2)


@jit
def model_add_scalar(params, value):
  t_list, tree_def = tree_flatten(params)
  new_t_list = [t + value for t in t_list]
  return tree_unflatten(tree_def, new_t_list)


@jit
def model_multiply_scalar(params, factor):
  t_list, tree_def = tree_flatten(params)
  new_t_list = [t * factor for t in t_list]
  return tree_unflatten(tree_def, new_t_list)


@jit
def model_average(params_list, weights=None):
  def average_fn(*tensor_list):
    return jnp.average(jnp.asarray(tensor_list), axis=0, weights=weights)

  return tree_map(average_fn, *params_list)


@jit
def model_flatten(params):
  tensors, tree_def = tree_flatten(params)
  flat_vec = jnp.concatenate([t.reshape(-1) for t in tensors])
  return flat_vec


@jit
def model_unflatten(flat_vec, params_template):
  t_list, tree_def = tree_flatten(params_template)
  pointer, split_list = 0, []
  for tensor in t_list:
    length = np.prod(tensor.shape)  # Shape is static so np is fine
    split_vec = flat_vec[pointer:pointer + length]
    split_list.append(split_vec.reshape(tensor.shape))
    pointer += length
  return tree_unflatten(tree_def, split_list)


@jit
def model_concat(params_list):
  flat_vecs = [model_flatten(params) for params in params_list]
  return jnp.concatenate(flat_vecs)


@jit
def model_zeros_like(params):
  return tree_map(jnp.zeros_like, params)

#######################################
############## Opt utils ##############
#######################################

def linf_proj_matrix(B, value=1):
  """
  Project a matrix onto an l_inf ball.
  """
  B_proj = jax.vmap(lambda X: projection.projection_l1_sphere(X, value=value), 
                    in_axes=0, 
                    out_axes=0)(B)
  return B_proj

def projection_linf_params(params, model_name, Bname = "linear_z"):

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