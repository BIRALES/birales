import os

# Try importing cupy for jit compilation onto GPU
cuda = None
try:
    from numba import cuda
except ModuleNotFoundError:
    pass


def try_cuda_jit(*jit_args, **jit_kwargs):
    def wrapped_jit(fn):
        if cuda is not None:
            try:
                return cuda.jit(*jit_args, **jit_kwargs)(fn)
            except Exception:
                pass
        else:
            return fn
    return wrapped_jit

