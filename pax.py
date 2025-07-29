# Pax -- JAX without the violence
# A simple, flax-like framework for working with JAX

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple

class Layer(ABC):
    @abstractmethod
    def weights(self, key: jax.Array): ...
    
    @abstractmethod
    def state(self): ...
    
    @abstractmethod
    def func(self) -> Callable[[dict, dict, jax.Array], Tuple[jax.Array, dict]]:
        """returns (params, state, x, *, is_training, key) -> (y, new_state)"""
        ...

@dataclass(frozen=True)
class Dense(Layer):
    in_nodes: int
    out_nodes: int
    activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    def __post_init__(self):
        # frozen dataclass hack: object.__setattr__
        if self.activation is None:
            object.__setattr__(self, "activation", lambda x: x)

    def weights(self, key) -> dict:
        k1, k2 = jax.random.split(key)
        w = jax.random.normal(k1, (self.in_nodes, self.out_nodes)) * jnp.sqrt(2. / self.in_nodes) # kaiming init
        b = jnp.zeros((self.out_nodes,))
        return {"w": w, "b": b}

    def state(self) -> dict:
        return {}

    def func(self) -> Callable[[dict, dict, jax.Array], Tuple[jax.Array, dict]]:
        def apply(params, state, x, *, is_training, key):
            y = self.activation(x @ params["w"] + params["b"])            
            return y, state

        return apply

@dataclass(frozen=True)
class Conv2D(Layer):
    size: int
    in_channels: int
    out_channels: int
    stride: int
    padding: str
    activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    def __post_init__(self):
        # frozen dataclass hack: object.__setattr__
        if self.activation is None:
            object.__setattr__(self, "activation", lambda x: x)

    def weights(self, key) -> dict:
        k1, k2 = jax.random.split(key)
        std = jnp.sqrt(2 / (self.in_channels*self.size*self.size))
        w = std*jax.random.normal(k1, (self.out_channels, self.in_channels, self.size, self.size))
        b = jnp.zeros((self.out_channels,)) # one per out_channel
        return dict(w=w, b=b)

    def state(self) -> dict:
        return {}

    def func(self) -> Callable[[dict, dict, jax.Array], Tuple[jax.Array, dict]]:
        def apply(params, state, x, *, is_training, key):
            y = jax.lax.conv(x, params['w'], window_strides=(self.stride, self.stride), padding=self.padding) + params['b'][:, None, None]
            y = self.activation(y)
            return y, state

        return apply

@dataclass(frozen=True)
class GlobalAvgPool(Layer):
    # feature_axes: tuple

    def weights(self, key):
        return {}

    def state(self):
        return {}

    def func(self) -> Callable[[dict, dict, jax.Array], Tuple[jax.Array, dict]]:
        def apply(params, state, x, *, is_training, key):
            return jnp.mean(x, axis=(2, 3), keepdims=True), state
            
        return apply

@dataclass(frozen=True)
class Flatten(Layer):
    def weights(self, key):
        return {}

    def state(self):
        return {}

    def func(self) -> Callable[[dict, dict, jax.Array], Tuple[jax.Array, dict]]:
        def apply(params, state, x, *, is_training, key):
            return x.reshape(x.shape[0], -1), {}

        return apply

@dataclass(frozen=True)
class BatchNorm(Layer):
    shape: tuple # data tensor shape
    feature_axes: tuple # feature axes for shape
    momentum: float = 0.1
    eps: float = 1e-5
    activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    
    def __post_init__(self) -> dict:
        if self.activation is None:
            object.__setattr__(self, "activation", lambda x: x)

    def weights(self, key) -> dict:
        # k1, k2 = jax.random.key(key)
        gamma, beta = jnp.ones((1,)+self.shape), jnp.zeros((1,)+self.shape)
        return dict(gamma=gamma, beta=beta)

    def state(self) -> dict:
        rmean, rvar = jnp.zeros((1,)+self.shape), jnp.ones((1,)+self.shape) # running mean and variance
        return dict(rmean=rmean, rvar=rvar) 

    def func(self) -> Callable[[dict, dict, jax.Array], Tuple[jax.Array, dict]]:
        def apply(params, state, x, *, is_training, key):
            rmean, rvar = state['rmean'], state['rvar']
            gamma, beta = params['gamma'], params['beta']
            
            reduce_axes = (0,) + tuple(         # always include batch axis 0
                d+1                             # offset into full‑tensor indices
                for d in range(x.ndim - 1)      # iterate over data dims
                if d not in self.feature_axes        # exclude the feature dims
            )
            bnmean = jnp.mean(x, axis=reduce_axes, keepdims=True)
            bnvar = jnp.var(x, axis=reduce_axes, keepdims=True)
            if is_training:
                rmean = rmean * (1 - self.momentum) + bnmean * self.momentum
                rvar = rvar * (1 - self.momentum) + bnvar * self.momentum
            else:
                bnmean = rmean
                bnvar = rvar
            x = (x - bnmean) / jnp.sqrt(bnvar + self.eps)
            y = x * gamma + beta
            y = self.activation(y)
            return y, dict(rmean=rmean, rvar=rvar)

        return apply
                
# def B_3_3(init_stride, in_size, out_size, in_channels, out_channels):
#     bn1 = BatchNorm((in_channels, in_size, in_size), feature_axes=(0,), activation=jax.nn.relu)
#     bn2 = BatchNorm((out_channels, out_size, out_size), feature_axes=(0,), activation=jax.nn.relu)

#     conv1 = Conv2D(size=3, stride=init_stride, in_channels=in_channels, out_channels = out_channels)
#     conv2 = Conv2D(size=3, stride=1, in_channels=out_channels, out_channels=out_channels)
#     c_skip = Conv2D(size=1, stride=init_stride, in_channels=in_channels, out_channels=out_channels)

#     return [bn1, conv1, bn2, conv2, c_skip]

@dataclass(frozen=True)
class B33(Layer):
    init_stride: int
    in_size: int
    out_size: int
    in_channels: int
    out_channels: int

    # pre-instantiate sublayers
    bn1: BatchNorm = field(init=False)
    bn2: BatchNorm = field(init=False)
    conv1: Conv2D = field(init=False)
    conv2: Conv2D = field(init=False)
    cskip: Conv2D = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "bn1", BatchNorm((self.in_channels, self.in_size, self.in_size),
                                                  feature_axes=(0,), activation=jax.nn.relu))
        object.__setattr__(self, "bn2", BatchNorm((self.out_channels, self.out_size, self.out_size),
                                                  feature_axes=(0,), activation=jax.nn.relu))
        object.__setattr__(self, "conv1", Conv2D(3, self.in_channels, self.out_channels, stride=self.init_stride, padding="SAME"))
        object.__setattr__(self, "conv2", Conv2D(3, self.out_channels, self.out_channels, stride=1, padding='SAME'))
        object.__setattr__(self, "cskip", Conv2D(1, self.in_channels, self.out_channels, stride=self.init_stride, padding='SAME'))


    def weights(self, key):
        keys = jax.random.split(key, 5)
        return {
            "bn1": self.bn1.weights(keys[0]),
            "conv1": self.conv1.weights(keys[1]),
            "bn2": self.bn2.weights(keys[2]),
            "conv2": self.conv2.weights(keys[3]),
            "cskip": self.cskip.weights(keys[4])
        }
    
    def state(self):
        return {
            "bn1": self.bn1.state(),
            "bn2": self.bn2.state(),
        }
    
    def func(self):
        def apply(params, state, x, *, is_training, key):
            residual = x
            
            key, sub = jax.random.split(key)
            x, bn1_state = self.bn1.func()(params["bn1"], state["bn1"], x, is_training=is_training, key=sub)
    
            key, sub = jax.random.split(key)
            x, _ = self.conv1.func()(params["conv1"], {}, x, is_training=is_training, key=sub)
    
            key, sub = jax.random.split(key)
            x, bn2_state = self.bn2.func()(params["bn2"], state["bn2"], x, is_training=is_training, key=sub)
    
            key, sub = jax.random.split(key)
            x, _ = self.conv2.func()(params["conv2"], {}, x, is_training=is_training, key=sub)
    
            # skip branch
            skip, _ = self.cskip.func()(params["cskip"], {}, x=residual, is_training=is_training, key=key)
    
            return x + skip, {"bn1": bn1_state, "bn2": bn2_state}
        return apply













