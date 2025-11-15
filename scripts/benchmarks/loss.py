import jax


from functools import partial

import torch
import auraloss
import korvax
import click

from time import time

from sot import Wasserstein1DLoss


@jax.jit
def jax_mrstft_grads(x, y):
    return jax.grad(jax_mrstft, argnums=0)(x, y)


@jax.jit
def jax_mrstft(x, y):
    return korvax.loss.mrstft_loss(
        x,
        y,
        hop_lengths=(32, 64, 128, 256, 512, 1024),
        win_lengths=(64, 128, 256, 512, 1024, 2048),
        fft_sizes=(128, 256, 512, 1024, 2048, 4096),
    )


torch_mrstft = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[128, 256, 512, 1024, 2048, 4096],
    hop_sizes=[32, 64, 128, 256, 512, 1024],
    win_lengths=[64, 128, 256, 512, 1024, 2048],
    w_sc=0,
    w_lin_mag=1,
    w_log_mag=1,
)

torch_mrstft = torch.compile(torch_mrstft)


jax_sot = jax.jit(
    partial(
        korvax.loss.time_frequency_loss,
        transform_fn=partial(
            korvax.spectrogram,
            hop_length=512,
            win_length=2048,
            window="flattop",
            power=2,
        ),
        loss_fn=partial(
            korvax.loss.spectral_optimal_transport_loss,
            balanced=True,
            normalize=True,
            limit_quantile_range=False,
            p=2,
        ),
    )
)

torch_sot = Wasserstein1DLoss(
    transform="stft",
    hop_length=512,
    fft_size=2048,
    window="flattop",
    square_magnitude=True,
    sample_rate=16000,
    balanced=True,
    normalize=True,
    quantile_lowpass=False,
    p=2,
)

torch_sot = torch.compile(torch_sot)


@jax.jit
def jax_sot_grads(x, y):
    return jax.grad(jax_sot, argnums=0)(x, y)


@torch.compile
def torch_sot_grads(x, y):
    loss = torch_sot(x, y)
    loss.backward()
    return x.grad


def run_jax_benchmark(fn, runs, batch_size, length, key):
    key, k1, k2 = jax.random.split(key, 3)
    x = jax.random.normal(key=k1, shape=(batch_size, 1, length))
    y = jax.random.normal(key=k2, shape=(batch_size, 1, length))
    out = fn(x, y).block_until_ready()  # noqa
    acc = 0
    for _ in range(runs):
        key, k1, k2 = jax.random.split(key, 3)
        x = jax.random.normal(key=k1, shape=(batch_size, 1, length))
        y = jax.random.normal(key=k2, shape=(batch_size, 1, length))
        start = time()
        out = fn(x, y).block_until_ready()  # noqa
        end = time()
        acc += end - start
    return acc / runs


def run_torch_benchmark(fn, runs, batch_size, length, device):
    x = torch.randn(size=(batch_size, 1, length), device=device)
    y = torch.randn(size=(batch_size, 1, length), device=device)
    out = fn(x, y)  # noqa
    if device == "cuda":
        torch.cuda.synchronize()
    acc = 0
    for _ in range(runs):
        x = torch.randn(size=(batch_size, 1, length), device=device)
        y = torch.randn(size=(batch_size, 1, length), device=device)
        start = time()
        out = fn(x, y)  # noqa
        if device == "cuda":
            torch.cuda.synchronize()
        end = time()
        acc += end - start
    return acc / runs


def run_torch_grad_benchmark(fn, runs, batch_size, length, device):
    x = torch.randn(size=(batch_size, 1, length), device=device, requires_grad=True)
    y = torch.randn(size=(batch_size, 1, length), device=device)
    fn(x, y)
    out = x.grad  # noqa
    if device == "cuda":
        torch.cuda.synchronize()
    acc = 0
    for _ in range(runs):
        x = torch.randn(size=(batch_size, 1, length), device=device, requires_grad=True)
        y = torch.randn(size=(batch_size, 1, length), device=device)
        start = time()
        fn(x, y)
        out = x.grad  # noqa
        if device == "cuda":
            torch.cuda.synchronize()
        end = time()
        acc += end - start
    return acc / runs


@torch.compile
def torch_mrstft_grads(x, y):
    loss = torch_mrstft(x, y)
    loss.backward()
    return x.grad


def run_benchmark(func, *args, runs: int = 100, **kwargs):
    func(*args, **kwargs)  # Warm-up
    start = time()
    for _ in range(runs):
        func(*args, **kwargs)
    end = time()
    return (end - start) / runs


@click.command()
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
@click.option("--batch-size", type=int, default=256)
@click.option("--length", type=int, default=64000)
@click.option("--seed", type=int, default=0)
@click.option("--runs", type=int, default=10)
def main(device, batch_size, length, seed, runs):
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    korvax_time = run_jax_benchmark(
        jax_mrstft, batch_size=batch_size, length=length, key=subkey, runs=runs
    )
    print(f"Korvax MRSTFT: {korvax_time * 1000:.3f} ms")

    torch_time = run_torch_benchmark(
        torch_mrstft, batch_size=batch_size, length=length, device=device, runs=runs
    )
    print(f"Torch MRSTFT: {torch_time * 1000:.3f} ms")

    key, subkey = jax.random.split(key)
    korvax_grad_time = run_jax_benchmark(
        jax_mrstft_grads, batch_size=batch_size, length=length, key=subkey, runs=runs
    )
    print(f"Korvax MRSTFT grads: {korvax_grad_time * 1000:.3f} ms")

    torch_grad_time = run_torch_grad_benchmark(
        torch_mrstft_grads,
        batch_size=batch_size,
        length=length,
        device=device,
        runs=runs,
    )
    print(f"Torch MRSTFT grads: {torch_grad_time * 1000:.3f} ms")

    key, subkey = jax.random.split(key)
    korvax_time = run_jax_benchmark(
        jax_sot, batch_size=batch_size, length=length, key=subkey, runs=runs
    )
    print(f"Korvax SOT: {korvax_time * 1000:.3f} ms")

    torch_time = run_torch_benchmark(
        torch_sot, batch_size=batch_size, length=length, device=device, runs=runs
    )
    print(f"Torch SOT: {torch_time * 1000:.3f} ms")

    key, subkey = jax.random.split(key)
    korvax_grad_time = run_jax_benchmark(
        jax_sot_grads, batch_size=batch_size, length=length, key=subkey, runs=runs
    )
    print(f"Korvax SOT grads: {korvax_grad_time * 1000:.3f} ms")

    torch_grad_time = run_torch_grad_benchmark(
        torch_sot_grads, batch_size=batch_size, length=length, device=device, runs=runs
    )
    print(f"Torch SOT grads: {torch_grad_time * 1000:.3f} ms")


if __name__ == "__main__":
    main()
