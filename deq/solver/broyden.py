import typing as tp

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

# from deqx import ad

class BroydenState(tp.NamedTuple):
    """
    Results from Broyden optimization.

    Parameters:
        converged: True if minimization converged.
        n_steps: integer the number of iterations of the BFGS update.
        min_x: array containing the minimum argument value found during the search. If
          the search converged, then this value is the argmin of the objective
          function.
        min_gx: array containing the value of the objective function at `min_x`. If the
          search converged, then this is the (local) minimum of the objective
          function.
        min_objective: array containing lowest 2 norm of the objective function
        x: array containing the prev argument value found during the search.
        gx: array containing the prev value of the objective function at `x`
        objective: array containing prev lowest 2 norm of the objective function
        trace: array of previous objectives
        Us: array containing the fraction component of the Jacobian approximation
            (N, 2d, L', n_step)
        VTs: array containing the \delta x_n^T J_{n-1}^{-1} of the estimated Jacobian
            (N, n_step, 2d, L')
        prot_break: True if protection threshold broken (no convergence)
        prog_break: True if progression threshold broken (no convergence)
    """

    converged: tp.Union[bool, jnp.ndarray]
    n_step: tp.Union[int, jnp.ndarray]
    min_x: jnp.ndarray
    min_gx: jnp.ndarray
    min_objective: jnp.ndarray
    x: jnp.ndarray
    gx: jnp.ndarray
    objective: jnp.ndarray
    trace: list
    Us: jnp.ndarray
    VTs: jnp.ndarray
    prot_break: tp.Union[bool, jnp.ndarray]
    prog_break: tp.Union[bool, jnp.ndarray]

    
_dot = partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)    
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)

def rmatvec(Us: jnp.ndarray, VTs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute `x.T @ (-I + U @ V.T)`.

    Args:
        Us: [p, d]
        VTs: [d, p]
        x: [p]

    Returns:
        out: [p]
    """
    if Us.size == 0:
        return -x
    xTU = _einsum("p, pd -> d", x, Us)  # (N, threshold)
    return -x + _einsum("d, dp -> p", xTU, VTs)  # (N, 2d, L'), but should really be (N, 1, (2d*L'))


def matvec(Us: jnp.ndarray, VTs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute `(-I + U @ V.T) @ x`.

    Args:
        Us: [p, d]
        VTs: [d, p]
        x: [p]

    Returns:
        out: [p]
    """
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # Us: (N, 2d, L', threshold)
    # VTs: (N, threshold, 2d, L')
    if Us.size == 0:
        return -x
    VTx = _einsum("dp, p -> d", VTs, x)  # (N, threshold)
    return -x + _einsum("pd, d -> p", Us, VTx)  # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def update(delta_x, delta_gx, Us, VTs, n_step):
    # Add column/row to Us/VTs with updated approximation
    # Calculate J_i
    vT = rmatvec(Us, VTs, delta_x)
    u = (delta_x - matvec(Us, VTs, delta_gx)) / jnp.dot(vT, delta_gx)

    vT = jnp.nan_to_num(vT)
    u = jnp.nan_to_num(u)

    # Store in UTs and VTs for calculating J 
    #VTs = jax.ops.index_update(VTs, jax.ops.index[n_step - 1], vT)
    VTs = VTs.at[n_step - 1].set(vT)
    #Us = jax.ops.index_update(Us, jax.ops.index[:, n_step - 1], u)
    Us = Us.at[:, n_step - 1].set(u)

    return Us, VTs


def line_search(g: tp.Callable,
                direction: jnp.ndarray,
                x0: jnp.ndarray,
                g0: jnp.ndarray,
                *args):
    """
    `update` is the proposed direction of update.
    """
    s = 1.0
    x_est = x0 + s * direction
    g0_new = g(x_est)
    return x_est - x0, g0_new - g0


class BroydenInfo(tp.NamedTuple):
    iterations: jnp.ndarray
    residual: jnp.ndarray
    diff_detail: jnp.ndarray
    prot_break: jnp.ndarray
    eps: jnp.ndarray
    
    
def broyden(g: tp.Callable,
            x0: jnp.ndarray,
            *args,
            maxiter: int = 50,
            eps: tp.Optional[float] = 1e-5,
            trace_size: int = 30) -> tp.Tuple[jnp.ndarray, BroydenInfo]:
    """
    Find the roots of g(x, *args), i.e. x_root s.t. g(x_root, *args) = 0.

    Args:
        g: Function to find roots of, e.g. g(x) = f(x) - x.
        x0: initial guess
        *args: used in g(x)
        maxiter: maximum number of iterations
        eps: terminates minimization when |J_inv| < eps.

    Returns:
        (x, info): x is approximate solution to fun(x, *args) == 0.
    """
    if eps is None:
        eps = 1e-6 * jnp.sqrt(x0.size)

    # Update rule is J <- J + delta_J,
    # so iteratively write J = J_0 + J_1 + J_2 + ...
    # For memory constraints J = U @ V^T
    # So J = U_0 @ V^T_0 + U_1 @ V^T_1 + ..
    # For fast calculation of inv_jacobian (approximately) we store as Us and VTs

    #bsz, total_hsize, seq_len = x0.shape
    x_shape = x0.shape
    x0 = jnp.reshape(x0, (-1,))
    param_size = x0.size

    def reshaped_fun(x):
        x = jnp.reshape(x, x_shape)
        gx = g(x, *args)
        return jnp.reshape(gx, (-1,))

    gx = reshaped_fun(x0)
    init_objective = jnp.linalg.norm(gx)

    # To be used in protective breaks
    trace = jnp.zeros(maxiter)
    #trace = jax.ops.index_update(trace, jax.ops.index[0], init_objective)
    trace = trace.at[0].set(init_objective)
    protect_thres = 1e5

    state = BroydenState(
        converged=False,
        n_step=0,
        min_x=x0,
        min_gx=gx,
        min_objective=init_objective,
        x=x0,
        gx=gx,
        objective=init_objective,
        trace=trace,
        Us=jnp.zeros((param_size, maxiter)),
        VTs=jnp.zeros((maxiter, param_size)),
        prot_break=False,
        prog_break=False,
    )

    def cond_fun(state: BroydenState):
        return (jnp.logical_not(state.converged)
                & jnp.logical_not(state.prot_break)
                & jnp.logical_not(state.prog_break)
                & (state.n_step < maxiter))

    def body_fun(state: BroydenState):
        inv_jacobian = -matvec(state.Us, state.VTs, state.gx)
        dx, delta_gx = line_search(reshaped_fun, inv_jacobian, state.x, state.gx, *args)

        state = state._replace(
            x = state.x + dx, 
            gx = state.gx + delta_gx,
            n_step = state.n_step + 1,
        )

        new_objective = jnp.linalg.norm(state.gx)
        #trace = jax.ops.index_update(
        #    state.trace, jax.ops.index[state.n_step % trace_size], new_objective
        #)
        trace = state.trace.at[state.n_step % trace_size].set(new_objective)

        min_found = new_objective < state.min_objective
        state = state._replace(
            # if a new minimum is found
            min_x = jnp.where(min_found, state.x, state.min_x),
            min_gx = jnp.where(min_found, state.gx, state.min_gx),
            min_objective = jnp.where(min_found, new_objective, state.min_objective),
            trace=trace,
            # check convergence
            converged = (new_objective < eps),
            prot_break = (new_objective > init_objective * protect_thres),
            prog_break = (new_objective < 3.0 * eps) & (state.n_step > trace_size) & (jnp.max(state.trace) / jnp.min(state.trace) < 1.3)
        )

        # update for next jacobian
        Us, VTs = update(dx, delta_gx, state.Us, state.VTs, state.n_step)
        state = state._replace(Us=Us, VTs=VTs)

        return state

    # state = body_fun(state)
    state = jax.lax.while_loop(cond_fun, body_fun, state)
    # state = jax.lax.fori_loop(0, maxiter, body_fun, state)
    # state = hk.fori_loop(0, maxiter, body_fun, state)
    result = jnp.reshape(state.min_x, x_shape)
  
    return {"x": result,
            "success": state.converged,
            "status": state.converged,
            "error": jnp.linalg.norm(state.min_gx),
            "fun": state.min_gx,
            "jac_inv": -matvec(state.Us, state.VTs, state.gx),
            #"diff_detail": jnp.linalg.norm(state.min_gx, axis=1),
            #"prot_break": state.prot_break,
            #"trace": state.trace,
            #"eps": eps,
            "nit": state.n_step
            }
    
    #result = state.min_x
    """
    info = BroydenInfo(state.n_step,
                       jnp.linalg.norm(state.min_gx),
                       jnp.abs(state.min_gx),
                       state.prot_break,
                       eps)

    return result, info

    """