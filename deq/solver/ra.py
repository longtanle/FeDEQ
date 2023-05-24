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
from deq.solver.fpi import FixedPointIteration
#from jaxopt._src.anderson import AndersonAcceleration
from deq.solver.anderson import AndersonAcceleration
from deq.solver.rootfinding_wrapper import BroydenRootFinding
from jaxopt._src import loop

class RAState(NamedTuple):
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
  s_param: Any
  aux: Optional[Any]


@dataclass(eq=False)
class RA(base.StochasticSolver):
  """Retrospective Approximation

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
  """
  optimality_fun: Callable
  value_and_grad: bool = False
  has_aux: bool = False
  avg: bool = False

  max_stepsize: float = 1.0

  fixed_point_solver: Callable = AndersonAcceleration
  pre_update: Optional[Callable] = None

  maxiter: int = 100
  maxls: int = 15
  tol: float = 1e-3
  verbose: int = 0

  implicit_diff: bool = False
  implicit_diff_solve: Optional[Callable] = None

  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self, init_params, solver = "anderson", *args, **kwargs) -> RAState:
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

    if solver == "anderson":
        self.fixed_point_solver = partial(AndersonAcceleration,
                                    history_size=5,
                                    ridge=1e-4,
                                    maxiter=20,
                                    tol=1e-2)

    elif solver == "broyden":
        self.fixed_point_solver = partial(BroydenRootFinding,
                                          tol = 1e-2)
                                
    else:
        self.fixed_point_solver = partial(FixedPointIteration,
                                    maxiter=20,
                                    tol=1e-2)

    self.solver = self.fixed_point_solver(fixed_point_fun = self.optimality_fun)  

    return RAState(iter_num=jnp.asarray(1),
                   error=jnp.asarray(jnp.inf),
                   value= self.optimality_fun(init_params, *args, **kwargs),
                   s_param = tree_zeros_like(init_params),
                   aux=aux)


  def update(self, params, state, *args, **kwargs) -> base.OptStep:
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

    next_params, solver_state = self.solver.run(params, *args, **kwargs)
    # Compute the estimate of the gradient

    # Update the estimate of the parameter
    s_param = tree_add(state.s_param, next_params)
    #theta_list = theta_list.at[k % averaged_iterates].set(theta) 

    #Averaged Iterates
    if self.avg:
      params_avg = tree_scalar_mul(1.0/state.iter_num, s_param)
    else:
      params_avg = next_params

    fun_value =  self.optimality_fun(params_avg, *args, **kwargs)

    # error of last step, avoid recomputing a gradient
    error = tree_l2_norm(fun_value, squared=False)

    next_state = RAState(iter_num=state.iter_num + 1,
                         error=error,
                         value=fun_value,
                         s_param = s_param,
                         aux=None)

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
