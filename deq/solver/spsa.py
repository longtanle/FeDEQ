# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGD solver with Armijo line search."""

from dataclasses import dataclass
from functools import partial

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp

from deq.solver.tree_util import tree_add_scalar_mul, tree_l2_norm, tree_average
from deq.solver.tree_util import tree_scalar_mul, tree_zeros_like, tree_random_bernoulli_like
from deq.solver.tree_util import tree_add, tree_sub, tree_mul, tree_div
from deq.solver import base
from jaxopt._src import loop

r"""The Simultaneous Perturbation Stochastic Approximation method (SPSA)
    is a stochastic approximation algorithm for optimizing cost functions whose evaluation may involve noise.

    While other gradient-based optimization methods usually attempt to compute
    the gradient analytically, SPSA involves approximating gradients at the cost of
    evaluating the cost function twice in each iteration step. This cost may result in
    a significant decrease in the overall cost of function evaluations for the entire optimization.
    It is based on an approximation of the unknown gradient :math:`\hat{g}(\hat{\theta}_{k})`
    through a simultaneous perturbation of the input parameters:

    .. math::
        \hat{g}_k(\hat{\theta}_k) = \frac{y(\hat{\theta}_k+c_k\Delta_k)-
        y(\hat{\theta}_k-c_k\Delta_k)}{2c_k} \begin{bmatrix}
           \Delta_{k1}^{-1} \\
           \Delta_{k2}^{-1} \\
           \vdots \\
           \Delta_{kp}^{-1}
         \end{bmatrix}\text{,}

    where

    * :math:`k` is the current iteration step,
    * :math:`\hat{\theta}_k` are the input parameters at iteration step :math:`k`,
    * :math:`y` is the objective function,
    * :math:`c_k=\frac{c}{k^\gamma}` is the gain sequence corresponding to evaluation step size
      and it can be controlled with

      * scaling parameter :math:`c` and
      * scaling exponent :math:`\gamma`

    * :math:`\Delta_{ki}^{-1} \left(1 \leq i \leq p \right)` are the inverted elements of
      random pertubation vector :math:`\Delta_k`.

    :math:`\hat{\theta}_k` is updated to a new set of parameters with

    .. math::
        \hat{\theta}_{k+1} = \hat{\theta}_{k} - a_k\hat{g}_k(\hat{\theta}_k)\text{,}

    where the gain sequences :math:`a_k=\frac{a}{(A+k)^\alpha}` controls parameter update step size.

    The gain sequence :math:`a_k` can be controlled with

    * scaling parameter :math:`a`,
    * scaling exponent :math:`\alpha` and
    * stability constant :math:`A`

    For more details, see `Spall (1998a)
    <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_.

    .. note::

        * One SPSA iteration step of a cost function that involves computing the expectation value of
          a Hamiltonian with ``M`` terms requires :math:`2*M` quantum device executions.
        * The forward-pass value of the cost function is not computed when stepping the optimizer.
          Therefore, in case of using ``step_and_cost`` method instead of ``step``, the number
          of executions will include the cost function evaluations.
"""

class SPSAState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number
    error: residuals of current estimate
    value: current value of the loss
    stepsize: current stepsize
    velocity: momentum term
  """
  iter_num: int
  error: float
  value: float
  s_ak: float
  s_param: Any
  aux: Optional[Any]
  stepsize: float


@dataclass(eq=False)
class SPSA(base.StochasticSolver):
  """Simultaneous perturbation Stochastic Approximation

  Attributes:
   fun: a smooth function of the form ``fun(parameters, *args, **kwargs)``,
      where ``parameters`` are the model parameters w.r.t. which we minimize
      the function and the rest are fixed auxiliary parameters.
    value_and_grad: whether ``fun`` just returns the value (False) or both
      the value and gradient (True).
    has_aux: whether ``fun`` outputs auxiliary data or not.
      If ``has_aux`` is False, ``fun`` is expected to be
        scalar-valued.
      If ``has_aux`` is True, then we have one of the following
        two cases.
      If ``value_and_grad`` is False, the output should be
      ``value, aux = fun(...)``.
      If ``value_and_grad == True``, the output should be
      ``(value, aux), grad = fun(...)``.
      At each iteration of the algorithm, the auxiliary outputs are stored
        in ``state.aux``.
    
    - theta: Initial function parameters (np.array)
    - n_iter: Number of iterations (int)
    - extra_params: Extra parameters taken by f (np.array)
    - theta_min: Minimum value of theta (np.array)
    - theta_max: Maximum value of theta (np.array)
    - constats: Constants needed for the gradient descent (dict)
    default is {"alpha": 0.602, "gamma": 0.101, "a": 0.6283185307179586, "c": 0.1, "A": False}


    maxiter: maximum number of solver iterations.
    maxls: maximum number of steps in line search.
    tol: tolerance to use.
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.

    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.

    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").

  References:
    For more details, see `Spall (1998a)
    <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_.
  """
  optimality_fun: Callable
  value_and_grad: bool = False
  has_aux: bool = False
  avg: bool = True

  alpha: float = 0.602  # default value recommended by the authors.
  gamma: float = 0.101  # default value recommended by the authors.
  a: float = 0.6283185307179586 # default value recommended by the authors.
  c: float = 0.1 # default value recommended by the authors.
  A: bool = False

  max_stepsize: float = 1.0

  pre_update: Optional[Callable] = None

  maxiter: int = 100
  maxls: int = 15
  tol: float = 1e-3
  verbose: int = 0

  implicit_diff: bool = False
  implicit_diff_solve: Optional[Callable] = None

  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self, init_params, *args, **kwargs) -> SPSAState:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to ``optimality_fun``.
      **kwargs: additional keyword arguments to be passed to ``optimality_fun``.
    Returns:
      state
    """

    if self.has_aux:
      _, aux = self.optimality_fun(init_params, *args, **kwargs)
    else:
      aux = None

    if self.A == False:
      self.A = self.maxiter / 10

    return SPSAState(iter_num=jnp.asarray(1),
                     error=jnp.asarray(jnp.inf),
                     value= self.optimality_fun(init_params, *args, **kwargs),
                     s_ak = jnp.asarray(0),
                     s_param = tree_zeros_like(init_params),
                     aux=aux,
                     stepsize = jnp.asarray(1.0))


  def update(self, key, params, state, *args, **kwargs) -> base.OptStep:
    """Performs one iteration of the solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to ``optimality_fun``.
      **kwargs: additional keyword arguments to be passed to ``optimality_fun``.
    Returns:
      (params, state)
    """

    #(value, aux), grad = self._value_and_grad_fun(params, *args, **kwargs)
    #grad_sqnorm = tree_l2_norm(grad, squared=True)
    if self.pre_update:
      params, state = self.pre_update(params, state, *args, **kwargs)

    # Update ak, ck  
    ak = self.a / (state.iter_num + self.A)**self.alpha
    s_ak = state.s_ak + ak
    #state.a_list = jnp.append(state.a_list, ak)  
    ck = self.c / state.iter_num**self.gamma



    # get the random perturbation vector from bernoulli distribution
    # it has to be symmetric around zero
    # But normal distribution does not work which makes the perturbations close to zero
    # Also, uniform distribution should not be used since they are not around zero
    #delta = 2 * jnp.round(jnp.random.rand(*params.shape)) - 1

    delta = tree_random_bernoulli_like(key, params)

    # Measure the loss function at perturbations
    #ck_delta = tree_scalar_mul(ck,delta)
    params_plus = tree_add_scalar_mul(params, ck, delta)
    params_minus = tree_add_scalar_mul(params, -ck, delta)

    y_plus = self.optimality_fun(params_plus, *args, **kwargs)
    y_minus = self.optimality_fun(params_minus, *args, **kwargs)

    # Compute the estimate of the gradient
    ck_delta = tree_scalar_mul(2.0*ck, delta)
    g_hat = tree_div(tree_sub(y_plus, y_minus), ck_delta)

    # Update the estimate of the parameter
    next_update = tree_scalar_mul(-ak, tree_mul(g_hat, self.optimality_fun(params, *args, **kwargs)))
    next_params = tree_add(params, next_update)
    s_param = tree_add(state.s_param, tree_scalar_mul(ak,next_params))
    #theta_list = theta_list.at[k % averaged_iterates].set(theta) 

    #Averaged Iterates
    if self.avg:
      params_avg = tree_scalar_mul(1.0/s_ak, s_param)
    else:
      params_avg = next_params
    fun_value =  self.optimality_fun(params_avg, *args, **kwargs)

    # error of last step, avoid recomputing a gradient
    error = tree_l2_norm(fun_value, squared=False)

    next_state = SPSAState(iter_num=state.iter_num + 1,
                           error=error,
                           value=fun_value,
                           s_ak = s_ak,
                           s_param = s_param,
                           aux=None,
                           stepsize=ak)

    return base.OptStep(next_params, next_state)

  # def optimality_fun(self, params, *args, **kwargs):
  #   """Optimality function mapping compatible with ``@custom_root``."""
  #   new_params, _ = self._fun(params, *args, **kwargs)
  #   return tree_sub(new_params, params)

  def __post_init__(self):
    if self.has_aux:
      self._fun = self.optimality_fun
    else:
      self._fun = lambda *a, **kw: (self.optimality_fun(*a, **kw), None)

    self.reference_signature = self.optimality_fun
