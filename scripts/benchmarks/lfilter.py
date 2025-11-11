import jax
import jax.numpy as jnp
import numpy as np

import scipy

from functools import partial

import torch
import torchaudio
import korvax
import click

from time import time


def torch_lfilter(b, a, x):
    return torchaudio.functional.lfilter(x, a, b, clamp=False)


def block_jax(fn, *args, **kwargs):
    return jax.tree.map(lambda x: x.block_until_ready(), fn(*args, **kwargs))


def jax_loss_fn(b, a, x):
    y = korvax.filter.lfilter(b, a, x, clamp=False)
    return jnp.mean(y**2)


def jax_grads(b, a, x):
    grads = jax.grad(jax_loss_fn, argnums=(0, 1))(b, a, x)
    return grads


def torch_loss_fn(b, a, x):
    y = torch_lfilter(b, a, x)
    return torch.mean(y**2)


def torch_grads(loss, b, a, x):
    b = b.requires_grad_(True)
    a = a.requires_grad_(True)
    loss = loss(b, a, x)
    loss.backward()
    return b.grad, a.grad


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
    k1, k2, k3 = jax.random.split(key, 3)

    if precision == "double":
        jax.config.update("jax_enable_x64", True)

    x = jax.random.normal(key=k1, shape=(batch_size, length))
    b = jax.random.normal(key=k2, shape=(order + 1,))
    a = jax.random.normal(key=k3, shape=(order + 1,))
    a = a.at[0].set((jnp.abs(a[0]) + 0.3).clip(max=1.0))

    dtype_np = np.float32 if precision == "single" else np.float64
    dtype_torch = torch.float32 if precision == "single" else torch.float64

    x_torch = torch.tensor(
        np.array(x), device=device, dtype=dtype_torch, requires_grad=False
    )
    a_torch = torch.tensor(
        np.array(a), device=device, dtype=dtype_torch, requires_grad=False
    )
    b_torch = torch.tensor(
        np.array(b), device=device, dtype=dtype_torch, requires_grad=False
    )

    x_np = np.array(x, dtype=dtype_np)
    a_np = np.array(a, dtype=dtype_np)
    b_np = np.array(b, dtype=dtype_np)

    korvax_jit = jax.jit(korvax.filter.lfilter, static_argnames=["clamp"])
    korvax_time = run_benchmark(
        partial(block_jax, korvax_jit), b, a, x, clamp=False, runs=runs
    )
    print(f"Korvax: {korvax_time * 1000:.3f} ms")

    torch_jit = torch.jit.trace(
        torch_lfilter,
        (b_torch, a_torch, x_torch),
    )
    torch_time = run_benchmark(torch_jit, b_torch, a_torch, x_torch, runs=runs)
    print(f"Torch: {torch_time * 1000:.3f} ms")

    if device == "cpu":
        scipy_time = run_benchmark(scipy.signal.lfilter, b_np, a_np, x_np, runs=runs)
        print(f"Scipy: {scipy_time * 1000:.3f} ms")

    korvax_grad_jit = jax.jit(jax_grads)
    korvax_grad_time = run_benchmark(
        partial(block_jax, korvax_grad_jit), b, a, x, runs=runs
    )
    print(f"Korvax grads: {korvax_grad_time * 1000:.3f} ms")

    b_torch.requires_grad_(True)
    a_torch.requires_grad_(True)

    torch_loss_jit = torch.jit.trace(torch_loss_fn, (b_torch, a_torch, x_torch))
    torch_grad_time = run_benchmark(
        partial(torch_grads, torch_loss_jit), b_torch, a_torch, x_torch, runs=runs
    )
    print(f"Torch grads: {torch_grad_time * 1000:.3f} ms")


if __name__ == "__main__":
    main()
