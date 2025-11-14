import jax
from .allpole import allpole as allpole
from . import _filter_cpu

jax.ffi.register_ffi_target("allpole", _filter_cpu.allpole_cpu(), platform="cpu")  # pyright: ignore[reportAttributeAccessIssue]
