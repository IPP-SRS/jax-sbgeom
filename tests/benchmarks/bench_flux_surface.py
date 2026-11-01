"""Warm (sync and pipelined) and compile timings of the FluxSurface scalars, positions and minimum_distance on the helias5 VMEC
surface (401 surfaces, 288 modes) and 50 helias5 coils. Note: on WSL2 GPUs a synchronised call is quantised to ~1.3 ms, so the
"pipelined" column (back-to-back dispatch, one sync) is the device time. Not collected by pytest.
Run: PYTHONPATH=src python tests/benchmarks/bench_flux_surface.py [--no-distance]"""
import argparse, os, time
import jax, jax.numpy as jnp, numpy as onp, equinox as eqx
jax.config.update("jax_enable_x64", True)
from jax_sbgeom.flux_surfaces import FluxSurface
from jax_sbgeom.distance import minimum_distance

p = argparse.ArgumentParser(); p.add_argument("--no-distance", action="store_true"); p.add_argument("--only", default="")
args = p.parse_args()
root = os.path.join(os.path.dirname(__file__), "..")
fs = FluxSurface.from_hdf5(os.path.join(root, "data/flux_surfaces/helias5_wout.h5"))
print(jax.devices()[0], "modes", fs.data.Rmnc.shape, flush=True)


def block(x):
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), x)


def bench(name, f, *a, reps=30):
    if args.only and args.only not in name:
        return
    jf = eqx.filter_jit(f)
    t0 = time.perf_counter(); comp = jf.lower(*a).compile(); tc = time.perf_counter() - t0
    for _ in range(5):
        block(comp(*a))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); block(comp(*a)); ts.append(time.perf_counter() - t0)
    tw = sorted(ts)[len(ts) // 2]
    t0 = time.perf_counter()
    for _ in range(reps):
        out = comp(*a)
    block(out)
    tp = (time.perf_counter() - t0) / reps
    print(f"{name:34s} sync {tw*1e3:9.2f} ms  pipelined {tp*1e3:9.2f} ms  compile {tc:7.2f} s", flush=True)


bench("volume", lambda f: f.volume(), fs)
bench("area", lambda f: f.area(), fs)
bench("mean_cross_section_area", lambda f: f.mean_cross_section_area(), fs)
bench("minor_radius", lambda f: f.minor_radius(), fs)
bench("major_radius", lambda f: f.major_radius(), fs)
bench("aspect_ratio", lambda f: f.aspect_ratio(), fs)
bench("five scalars (one jit)", lambda f: jnp.stack([f.major_radius(), f.minor_radius(), f.aspect_ratio(), f.volume(), f.area()]), fs)
th, ph = jnp.meshgrid(jnp.linspace(0, 2*jnp.pi, 60, endpoint=False), jnp.linspace(0, 2*jnp.pi, 100, endpoint=False), indexing="ij")
bench("cartesian_position 60x100", lambda f, t, q: f.cartesian_position(1.0, t, q), fs, th, ph)
bench("magnetic_axis-like (s=0, 256 phi)", lambda f, q: f.cartesian_position(0.0, jnp.zeros_like(q), q), fs, jnp.linspace(0, 2*jnp.pi, 256, endpoint=False))
bench("normal 60x100", lambda f, t, q: f.normal(1.0, t, q), fs, th, ph)

if not args.no_distance:
    from jax_sbgeom.coils import CoilSet, DiscreteCoil, convert_to_fourier_coilset
    v = onp.load(os.path.expanduser("~/full_coupling/functional_preprocess/tests/data/helias5_reference.npz"))["coil_vertices_nonperiodic_full"]
    coils = convert_to_fourier_coilset(CoilSet(DiscreteCoil.from_positions(jnp.asarray(v))), None).coils
    bench("minimum_distance 50 coils", lambda c, f: jax.vmap(lambda ci: minimum_distance(ci, f).distance)(c), coils, fs, reps=10)
