import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

import torch
import torchlpc
import jaxpole.filter
import korvax
import click

from time import time


# @torch.compile compilation throws a bunch of errors...
def torch_allpole(a, x):
    return torchlpc.sample_wise_lpc(x, a)


def block_jax(fn, *args, **kwargs):
    fn(*args, **kwargs).block_until_ready()


def jax_loss_fn(a, x):
    y = jax.vmap(korvax.filter.allpole)(x, a)
    return jnp.mean(y**2)


def jaxpole_loss_fn(a, x):
    zi = jnp.zeros((x.shape[0], a.shape[2]))
    y = jaxpole.filter.allpole(x, a, zi)
    return jnp.mean(y**2)


def jax_grads(a, x):
    grads = jax.grad(jax_loss_fn, argnums=0)(a, x)
    return grads


def torch_loss_fn(a, x):
    y = torch_allpole(a, x)
    return torch.mean(y**2)


def torch_grads(a, x):
    loss = torch_loss_fn(a, x)
    loss.backward()
    return a.grad


def run_benchmark(func, *args, runs: int = 100, **kwargs):
    func(*args, **kwargs)  # Warm-up
    start = time()
    for _ in range(runs):
        func(*args, **kwargs)
    end = time()
    return (end - start) / runs


@click.command()
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
@click.option("--batch-size", type=int, default=128)
@click.option("--order", type=int, default=16)
@click.option("--length", type=int, default=64000)
@click.option("--seed", type=int, default=0)
@click.option("--runs", type=int, default=100)
@click.option("--precision", type=click.Choice(["single", "double"]), default="single")
def main(device, batch_size, order, length, seed, runs, precision):
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)

    if precision == "double":
        jax.config.update("jax_enable_x64", True)

    x = jax.random.normal(key=k1, shape=(batch_size, length))
    a = jax.random.normal(key=k2, shape=(batch_size, length, order)) * 0.1

    dtype_torch = torch.float32 if precision == "single" else torch.float64

    x_torch = torch.tensor(
        np.array(x), device=device, dtype=dtype_torch, requires_grad=False
    )
    a_torch = torch.tensor(
        np.array(a), device=device, dtype=dtype_torch, requires_grad=False
    )

    korvax_jit = jax.jit(jax.vmap(korvax.filter.allpole))

    zi = jnp.zeros((batch_size, order))
    korvax_time = run_benchmark(partial(block_jax, korvax_jit), x, a, zi, runs=runs)
    print(f"Korvax: {korvax_time * 1000:.3f} ms")

    zi = jnp.zeros((batch_size, order))
    jaxpole_jit = jax.jit(jaxpole.filter.allpole, static_argnames=[])
    jaxpole_time = run_benchmark(partial(block_jax, jaxpole_jit), x, a, zi, runs=runs)
    print(f"JAXPole: {jaxpole_time * 1000:.3f} ms")

    torch_time = run_benchmark(torch_allpole, a_torch, x_torch, runs=runs)
    print(f"Torch: {torch_time * 1000:.3f} ms")

    korvax_grad_jit = jax.jit(jax_grads)
    korvax_grad_time = run_benchmark(
        partial(block_jax, korvax_grad_jit), a, x, runs=runs
    )
    print(f"Korvax grads: {korvax_grad_time * 1000:.3f} ms")

    jaxpole_grad_jit = jax.jit(jax.grad(jaxpole_loss_fn))
    jaxpole_grad_time = run_benchmark(
        partial(block_jax, jaxpole_grad_jit), a, x, runs=runs
    )
    print(f"JAXPole grads: {jaxpole_grad_time * 1000:.3f} ms")

    a_torch = a_torch.requires_grad_(True)

    torch_grad_time = run_benchmark(torch_grads, a_torch, x_torch, runs=runs)
    print(f"Torch grads: {torch_grad_time * 1000:.3f} ms")


if __name__ == "__main__":
    main()
