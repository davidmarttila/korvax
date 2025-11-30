import jax
import jax.numpy as jnp


import torch
import korvax
import click

from torchlpc import sample_wise_lpc
from time import time

sample_wise_lpc = torch.compile(sample_wise_lpc)


@jax.jit
def jax_grad(x, a):
    def loss_fn(x, a):
        y = jax.vmap(korvax.filter.ltv.allpole)(x, a)
        return jnp.mean(y**2)

    grad = jax.grad(loss_fn, argnums=1)(x, a)
    return grad


@torch.compile
def torch_grad(x, a):
    y = sample_wise_lpc(x, a)
    loss = torch.mean(y**2)  # pyright: ignore[reportOperatorIssue]
    loss.backward()


def jax_make_inputs(batch_size, length, order, key):
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(key=k1, shape=(batch_size, length, order)) * 0.1
    x = jax.random.normal(key=k2, shape=(batch_size, length))
    return a.block_until_ready(), x.block_until_ready()


def run_jax_grad_benchmark(runs, batch_size, length, order, key):
    key, subkey = jax.random.split(key, 2)
    a, x = jax_make_inputs(batch_size, length, order, subkey)
    out = jax_grad(x, a).block_until_ready()  # noqa
    acc = 0
    for _ in range(runs):
        key, subkey = jax.random.split(key, 2)
        a, x = jax_make_inputs(batch_size, length, order, subkey)
        start = time()
        out = jax_grad(x, a).block_until_ready()  # noqa
        end = time()
        acc += end - start
    return acc / runs


@jax.jit
def jax_value(x, a):
    return jax.vmap(korvax.filter.ltv.allpole)(x, a)


def run_jax_benchmark(runs, batch_size, length, order, key):
    key, subkey = jax.random.split(key, 2)
    a, x = jax_make_inputs(batch_size, length, order, subkey)
    out = jax_value(x, a).block_until_ready()  # noqa
    acc = 0
    for _ in range(runs):
        key, subkey = jax.random.split(key, 2)
        a, x = jax_make_inputs(batch_size, length, order, subkey)
        start = time()
        out = jax_value(x, a).block_until_ready()  # noqa
        end = time()
        acc += end - start
    return acc / runs


def run_torch_benchmark(runs, batch_size, length, order, device):
    a = torch.randn(size=(batch_size, length, order), device=device) * 0.1
    x = torch.randn(size=(batch_size, length), device=device)
    out = sample_wise_lpc(x, a)  # noqa
    acc = 0
    for _ in range(runs):
        a = torch.randn(size=(batch_size, length, order), device=device) * 0.1
        x = torch.randn(size=(batch_size, length), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time()
        out = sample_wise_lpc(x, a)  # noqa
        if device == "cuda":
            torch.cuda.synchronize()
        end = time()
        acc += end - start
    return acc / runs


def run_torch_grad_benchmark(runs, batch_size, length, order, device):
    a = torch.randn(size=(batch_size, length, order), device=device, requires_grad=True)
    b = a * 0.1
    x = torch.randn(size=(batch_size, length), device=device)
    torch_grad(x, b)
    out = a.grad  # noqa

    acc = 0
    for _ in range(runs):
        a = torch.randn(
            size=(batch_size, length, order), device=device, requires_grad=True
        )
        b = a * 0.1
        x = torch.randn(size=(batch_size, length), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time()
        torch_grad(x, b)
        out = a.grad  # noqa
        if device == "cuda":
            torch.cuda.synchronize()
        end = time()
        acc += end - start
    return acc / runs


@click.command()
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
@click.option("--batch-size", type=int, default=128)
@click.option("--order", type=int, default=16)
@click.option("--length", type=int, default=64000)
@click.option("--seed", type=int, default=0)
@click.option("--runs", type=int, default=100)
def main(device, batch_size, order, length, seed, runs):
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)

    torch_time = run_torch_benchmark(
        runs=runs, batch_size=batch_size, length=length, order=order, device=device
    )
    print(f"Torch: {torch_time * 1000:.3f} ms")

    korvax_time = run_jax_benchmark(
        runs=runs, batch_size=batch_size, length=length, order=order, key=k1
    )
    print(f"Korvax: {korvax_time * 1000:.3f} ms")

    torch_grad_time = run_torch_grad_benchmark(
        runs=runs, batch_size=batch_size, length=length, order=order, device=device
    )
    print(f"Torch grads: {torch_grad_time * 1000:.3f} ms")

    korvax_grad_time = run_jax_grad_benchmark(
        runs=runs, batch_size=batch_size, length=length, order=order, key=k2
    )
    print(f"Korvax grads: {korvax_grad_time * 1000:.3f} ms")


if __name__ == "__main__":
    main()
