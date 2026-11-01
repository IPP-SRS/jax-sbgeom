import jax
import optax
from functools import partial
import jax.numpy as jnp
from typing import Callable, Tuple
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class OptimizationSettings:
    '''
    Settings for optimization routines.

    Attributes
    ----------
    max_iterations : int
        Maximum number of iterations for the optimizer
    tolerance : float
        Tolerance for convergence   
    '''
    max_iterations : int
    tolerance      : float

@partial(jax.jit, static_argnums=(2,3,4))
def run_lbfgs_step(params : optax.Params, opt_state : optax.OptState, loss_fn : Callable[[optax.Params], float], optimizer : optax.GradientTransformationExtraArgs, value_and_grad_function :  Callable[[optax.Params], Tuple[float, optax.Params]]):    
    '''
    Run a single step of L-BFGS optimization.

    Parameters
    ----------
    params : optax.Params
        Current parameters for optimization
    opt_state : optax.OptState
        Current optimizer state
    loss_fn : Callable[[optax.Params], float]
        Loss function to minimize
    optimizer : optax.GradientTransformationExtraArgs
        L-BFGS optimizer
    value_and_grad_function : Callable[[optax.Params], Tuple[float, optax.Params]]
        Function to compute value and gradient of the loss function
    Returns
    -------
    optax.Params
        Updated parameters after the optimization step
    optax.OptState
        Updated optimizer state after the optimization step
    '''
    value, grad = value_and_grad_function(params, state=opt_state)
    updates, opt_state = optimizer.update(
            grad, opt_state, params=params, value=value, grad=grad, value_fn=loss_fn
        )
    params = optax.apply_updates(params, updates)
    return params, opt_state


_LBFGS = optax.lbfgs()      # stateless, one module-level instance


def _call_loss(loss_fn, params, args):
    return loss_fn(params) if args is None else loss_fn(params, args)


def _lbfgs_continue(carry):
    _, state, _, max_iterations, tolerance = carry
    iter_num = optax.tree.get(state, 'count')
    grad     = optax.tree.get(state, 'grad')
    err      = optax.tree.norm(grad)
    return (iter_num == 0) | ((iter_num < max_iterations) & (err >= tolerance))


def _lbfgs_body(loss_fn, carry):
    params, state, args, max_iterations, tolerance = carry
    loss   = partial(_call_loss, loss_fn)
    value, grad = optax.value_and_grad_from_state(loss)(params, args, state=state)
    updates, state = _LBFGS.update(grad, state, params=params, value=value, grad=grad, value_fn=loss, args=args)
    return optax.apply_updates(params, updates), state, args, max_iterations, tolerance


@partial(jax.jit, static_argnums=(1, 2))
def run_optimization_lbfgs(initial_values : optax.Params, loss_fn : Callable[..., float], settings : OptimizationSettings, args=None):
    '''
    Minimise a loss with L-BFGS.

    Parameters
    ----------
    initial_values : optax.Params
        Initial parameters for optimization
    loss_fn : Callable
        Loss function to minimize: loss_fn(params) or, if `args` is given, loss_fn(params, args). Static (jit): pass a
        module-level function and put everything it needs in `args`, so that repeated calls hit the trace cache.
    settings : OptimizationSettings
        Settings for the optimization
    args : pytree, optional
        Traced data passed to loss_fn
    Returns
    -------
    (optax.Params, optax.OptState)
        Optimized parameters and the final optimizer state
    '''
    carry = (initial_values, _LBFGS.init(initial_values), args, jnp.asarray(settings.max_iterations), jnp.asarray(settings.tolerance))
    params, state, _, _, _ = jax.lax.while_loop(_lbfgs_continue, partial(_lbfgs_body, loss_fn), carry)
    return params, state
