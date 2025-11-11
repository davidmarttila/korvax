from jaxtyping import ArrayLike, Float

_WindowSpec = str | float | tuple | Float[ArrayLike, " win_length"]
