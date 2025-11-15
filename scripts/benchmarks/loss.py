import jax
import numpy as np


from functools import partial

import torch
import auraloss
import korvax
import click

from time import time

from sot import Wasserstein1DLoss


def block_jax(fn, *args, **kwargs):
    fn(*args, **kwargs).block_until_ready()


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


# this made performance 4x worse!!
# torch._dynamo.config.compiled_autograd = True


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
@click.option("--precision", type=click.Choice(["single", "double"]), default="single")
def main(device, batch_size, length, seed, runs, precision):
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)

    if precision == "double":
        jax.config.update("jax_enable_x64", True)

    x = jax.random.normal(key=k1, shape=(batch_size, 1, length))
    y = jax.random.normal(key=k2, shape=(batch_size, 1, length))

    dtype_torch = torch.float32 if precision == "single" else torch.float64

    x_torch = torch.tensor(
        np.array(x), device=device, dtype=dtype_torch, requires_grad=False
    )
    y_torch = torch.tensor(
        np.array(y), device=device, dtype=dtype_torch, requires_grad=False
    )

    korvax_time = run_benchmark(partial(block_jax, jax_mrstft), x, y, runs=runs)
    print(f"Korvax MRSTFT: {korvax_time * 1000:.3f} ms")

    torch_time = run_benchmark(torch_mrstft, x_torch, y_torch, runs=runs)
    print(f"Torch MRSTFT: {torch_time * 1000:.3f} ms")

    korvax_grad_time = run_benchmark(
        partial(block_jax, jax_mrstft_grads), x, y, runs=runs
    )
    print(f"Korvax MRSTFT grads: {korvax_grad_time * 1000:.3f} ms")
    x_torch = x_torch.requires_grad_(True)

    torch_grad_time = run_benchmark(torch_mrstft_grads, x_torch, y_torch, runs=runs)
    print(f"Torch MRSTFT grads: {torch_grad_time * 1000:.3f} ms")

    korvax_time = run_benchmark(partial(block_jax, jax_sot), x, y, runs=runs)
    print(f"Korvax SOT: {korvax_time * 1000:.3f} ms")

    torch_time = run_benchmark(torch_sot, x_torch, y_torch, runs=runs)
    print(f"Torch SOT: {torch_time * 1000:.3f} ms")

    korvax_grad_time = run_benchmark(partial(block_jax, jax_sot_grads), x, y, runs=runs)
    print(f"Korvax SOT grads: {korvax_grad_time * 1000:.3f} ms")
    x_torch = x_torch.requires_grad_(True)

    torch_grad_time = run_benchmark(torch_sot_grads, x_torch, y_torch, runs=runs)
    print(f"Torch SOT grads: {torch_grad_time * 1000:.3f} ms")


if __name__ == "__main__":
    main()
