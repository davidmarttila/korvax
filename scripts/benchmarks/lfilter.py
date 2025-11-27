import jax
import jax.numpy as jnp

import torch
import click

from time import time

from philtorch.lti import lfilter as phil_lfilter

from korvax.filter import lfilter as korvax_lfilter
from torchaudio.functional import lfilter as torch_lfilter

# torch_lfilter = torch.compile(torch_lfilter)
phil_lfilter = torch.compile(phil_lfilter, backend="diag_ssm")


@jax.jit
def jax_grad(x, a, b):
    def loss_fn(x, a, b):
        y = jax.vmap(korvax_lfilter)(x, a, b)
        return jnp.mean(y**2)

    grad = jax.grad(loss_fn, argnums=(1, 2))(x, a, b)
    return grad


# @torch.compile
def torch_grad(x, a, b):
    y = torch_lfilter(x, a, b)
    loss = torch.mean(y**2)
    loss.backward()


@torch.compile
def phil_grad(x, a, b):
    y = phil_lfilter(b, a, x)
    loss = torch.mean(y**2)
    loss.backward()


def jax_make_inputs(batch_size, length, order, key):
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(key=k1, shape=(batch_size, order)) * 0.1
    b = jax.random.normal(key=k2, shape=(batch_size, order)) * 0.1
    x = jax.random.normal(key=k2, shape=(batch_size, length))
    return x.block_until_ready(), a.block_until_ready(), b.block_until_ready()


def run_jax_grad_benchmark(runs, batch_size, length, order, key):
    key, subkey = jax.random.split(key, 2)
    x, a, b = jax_make_inputs(batch_size, length, order, subkey)
    a_grad, b_grad = jax_grad(x, a, b)
    a_grad.block_until_ready()
    b_grad.block_until_ready()
    acc = 0
    for _ in range(runs):
        key, subkey = jax.random.split(key, 2)
        x, a, b = jax_make_inputs(batch_size, length, order, subkey)
        start = time()
        a_grad, b_grad = jax_grad(x, a, b)
        a_grad.block_until_ready()
        b_grad.block_until_ready()
        end = time()
        acc += end - start
    return acc / runs


@jax.jit
def jax_value(x, a, b):
    return jax.vmap(korvax_lfilter)(x, a, b)


def run_jax_benchmark(runs, batch_size, length, order, key):
    key, subkey = jax.random.split(key, 2)
    x, a, b = jax_make_inputs(batch_size, length, order, subkey)
    out = jax_value(x, a, b).block_until_ready()  # noqa
    acc = 0
    for _ in range(runs):
        key, subkey = jax.random.split(key, 2)
        x, a, b = jax_make_inputs(batch_size, length, order, subkey)
        start = time()
        out = jax_value(x, a, b).block_until_ready()  # noqa
        end = time()
        acc += end - start
    return acc / runs


def run_torch_benchmark(runs, batch_size, length, order, device):
    a = torch.randn(size=(batch_size, order), device=device) * 0.1
    b = torch.randn(size=(batch_size, order), device=device) * 0.1
    x = torch.randn(size=(batch_size, length), device=device)
    out = torch_lfilter(x, a, b)  # noqa
    acc = 0
    for _ in range(runs):
        a = torch.randn(size=(batch_size, order), device=device) * 0.1
        b = torch.randn(size=(batch_size, order), device=device) * 0.1
        x = torch.randn(size=(batch_size, length), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time()
        out = torch_lfilter(x, a, b)  # noqa
        if device == "cuda":
            torch.cuda.synchronize()
        end = time()
        acc += end - start
    return acc / runs


def run_torch_grad_benchmark(runs, batch_size, length, order, device):
    a = torch.randn(size=(batch_size, order), device=device, requires_grad=True)
    b = torch.randn(size=(batch_size, order), device=device, requires_grad=True)
    x = torch.randn(size=(batch_size, length), device=device)
    torch_grad(x, a * 0.1, b * 0.1)
    out = a.grad  # noqa

    acc = 0
    for _ in range(runs):
        a = torch.randn(size=(batch_size, order), device=device, requires_grad=True)
        b = torch.randn(size=(batch_size, order), device=device, requires_grad=True)
        a_ = a * 0.1
        b_ = b * 0.1

        x = torch.randn(size=(batch_size, length), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time()
        torch_grad(x, a_, b_)
        out = a.grad, b.grad  # noqa
        if device == "cuda":
            torch.cuda.synchronize()
        end = time()
        acc += end - start
    return acc / runs


def run_phil_benchmark(runs, batch_size, length, order, device):
    a = torch.randn(size=(batch_size, order), device=device)
    b = torch.randn(size=(batch_size, order), device=device)
    x = torch.randn(size=(batch_size, length), device=device)
    out = phil_lfilter(b * 0.1, a * 0.1, x)  # noqa
    acc = 0
    for _ in range(runs):
        a = torch.randn(size=(batch_size, order), device=device)
        b = torch.randn(size=(batch_size, order), device=device)
        a_ = a * 0.1
        b_ = b * 0.1
        x = torch.randn(size=(batch_size, length), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time()
        out = phil_lfilter(b_, a_, x)  # noqa
        if device == "cuda":
            torch.cuda.synchronize()
        end = time()
        acc += end - start
    return acc / runs


def run_phil_grad_benchmark(runs, batch_size, length, order, device):
    a = torch.randn(size=(batch_size, order), device=device, requires_grad=True)
    b = torch.randn(size=(batch_size, order), device=device, requires_grad=True) * 0.1
    x = torch.randn(size=(batch_size, length), device=device)
    phil_grad(x, a, b)
    out = a.grad  # noqa

    acc = 0
    for _ in range(runs):
        a = torch.randn(size=(batch_size, order), device=device, requires_grad=True)
        b = torch.randn(size=(batch_size, order), device=device, requires_grad=True)
        a_ = a * 0.1
        b_ = b * 0.1
        x = torch.randn(size=(batch_size, length), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time()
        phil_grad(x, a_, b_)
        out = a.grad, b.grad  # noqa
        if device == "cuda":
            torch.cuda.synchronize()
        end = time()
        acc += end - start
    return acc / runs


@click.command()
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
@click.option("--batch-size", type=int, default=128)
@click.option("--order", type=int, default=4)
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

    phil_time = run_phil_benchmark(
        runs=runs, batch_size=batch_size, length=length, order=order, device=device
    )
    print(f"Phil: {phil_time * 1000:.3f} ms")

    korvax_time = run_jax_benchmark(
        runs=runs, batch_size=batch_size, length=length, order=order, key=k1
    )
    print(f"Korvax: {korvax_time * 1000:.3f} ms")

    torch_grad_time = run_torch_grad_benchmark(
        runs=runs, batch_size=batch_size, length=length, order=order, device=device
    )
    print(f"Torch grads: {torch_grad_time * 1000:.3f} ms")

    phil_grad_time = run_phil_grad_benchmark(
        runs=runs, batch_size=batch_size, length=length, order=order, device=device
    )
    print(f"Phil grads: {phil_grad_time * 1000:.3f} ms")

    korvax_grad_time = run_jax_grad_benchmark(
        runs=runs, batch_size=batch_size, length=length, order=order, key=k2
    )
    print(f"Korvax grads: {korvax_grad_time * 1000:.3f} ms")


if __name__ == "__main__":
    main()
