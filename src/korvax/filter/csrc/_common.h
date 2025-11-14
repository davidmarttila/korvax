#include "nanobind/nanobind.h"

#define XLA_CHECK_ARG(cond, message) \
    if (!(cond))                     \
        return ffi::Error::InvalidArgument(message);

template <typename T>
nanobind::capsule EncapsulateFfiCall(T *fn)
{
    // This check is optional, but it can be helpful for avoiding invalid handlers.
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be an XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void *>(fn));
}