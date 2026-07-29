"""jax-sbgeom: A JAX-based package for geometric operations."""

from importlib.metadata import version, PackageNotFoundError

from . import flux_surfaces, jax_utils, coils, interfaces

try:
    __version__ = version("jax-sbgeom")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Import main modules here as the package grows
__all__ = ["__version__", "flux_surfaces", "jax_utils", "coils", "interfaces"]
