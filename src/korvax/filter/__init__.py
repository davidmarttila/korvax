import jax
from . import _filter_cpu
from .allpole import allpole as allpole

jax.ffi.register_ffi_target("allpole", _filter_cpu.allpole_cpu(), platform="cpu")  # pyright: ignore[reportAttributeAccessIssue]
