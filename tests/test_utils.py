import pytest
import korvax
import jax.numpy as jnp


@pytest.mark.parametrize(
    "size,frame_length,hop_length",
    [
        (11, 4, 2),
        ((2, 11), 4, 2),
        ((3, 2, 11), 4, 1),
        ((2, 3, 11), 5, 6),
    ],
)
def test_frame(size, frame_length, hop_length):
    x = jnp.ones(size)
    korvax.util.frame(x, frame_length=frame_length, hop_length=hop_length)


def test_pad_center():
    x = jnp.ones((2, 11))
    korvax.util.pad_center(x, size=15)
