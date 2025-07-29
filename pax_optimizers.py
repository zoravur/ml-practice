# Pax -- JAX without the violence
# A simple, flax-like framework for working with JAX

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple

@dataclass(frozen=True)
class Adam:
    lr: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    def init(self, params):
        m = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params)
        t = jnp.array(0)
        return dict(m=m, v=v, t=t)

    def update(self, grads, state, lr, params=None):
        t = state["t"] + 1
        m = jax.tree.map(lambda m, g: self.b1*m + (1-self.b1)*g, state["m"], grads)
        v = jax.tree.map(lambda v, g: self.b2*v + (1-self.b2)*(g*g), state["v"], grads)
        m_hat = jax.tree.map(lambda m: m / (1 - self.b1**t), m)
        v_hat = jax.tree.map(lambda v: v / (1 - self.b2**t), v)
        updates = jax.tree.map(lambda m, v: -lr * m / (jnp.sqrt(v) + self.eps), m_hat, v_hat)
        return updates, dict(m=m, v=v, t=t)

    def apply_updates(self, params, updates):
        return jax.tree.map(lambda p, u: p + u, params, updates)



