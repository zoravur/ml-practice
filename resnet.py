import jax
from pax import Dense, Conv2D, BatchNorm, B33


def wideblock(*, init_stride, out_size, in_channels, out_channels):
    b33_1 = B33(init_stride=init_stride, in_size=out_size*init_stride, out_size=out_size, in_channels=in_channels, out_channels=out_channels)
    return b33_1 + [B33(1, out_size, out_channels, out_channels)] * 3
