from functools import partial

import jax
import jax.numpy as jnp

import numpy as onp


# Allowed closure (factory called at import time, never per call; the result has a stable identity).
def stack_jacfwd(fun, argnums):
    jacfwd_internal = jax.jacfwd(fun, argnums = argnums)
    def jac_stack_wrap(*args):                
        return jnp.stack(jacfwd_internal(*args), axis=-1)    
    return jac_stack_wrap


def tree_take(tree, idx):
    '''`x[idx]` for every leaf x of `tree` (no per-call closure, unlike `tree_map(lambda x: x[idx], tree)`).'''
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return jax.tree_util.tree_unflatten(treedef, [x[idx] for x in leaves])


def _apply(fn, consts, x):
    return fn(x, consts)


def map_chunked(fn, xs, consts, batch_size):
    '''
    `fn(x, consts)` for every x of the pytree `xs` along its leading axis [n]: `jax.lax.map` with `batch_size` entries vectorised
    (vmap) per chunk and the chunks sequential, so a chunk's intermediates bound the forward memory. Each chunk is checkpointed,
    so the backward pass recomputes one chunk at a time instead of storing the intermediates of all chunks.
    `batch_size` is a static int >= 1; batch_size = n evaluates everything at once.
    `fn` is a module-level function; `consts` (pytree) is passed unchanged.
    '''
    return jax.lax.map(jax.checkpoint(partial(_apply, fn, consts), prevent_cse=False), xs, batch_size=batch_size)
