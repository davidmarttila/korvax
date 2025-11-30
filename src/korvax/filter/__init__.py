import jax
from . import lti as lti
from . import ltv as ltv
from . import _filter_cpu

try:
    from . import _filter_gpu
except ImportError:
    _filter_gpu = None

jax.ffi.register_ffi_target("allpole_cpu", _filter_cpu.allpole(), platform="cpu")  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue]
if _filter_gpu is not None:
    jax.ffi.register_ffi_target("allpole_cuda", _filter_gpu.allpole(), platform="CUDA")  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue]
