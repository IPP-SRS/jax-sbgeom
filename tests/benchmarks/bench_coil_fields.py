"""Note: on WSL2 GPUs a synchronised call has a ~1.3 ms granularity (any kernel > ~0.2 ms is reported as 1.3 ms), so the
"pipelined" column (device time) is the meaningful one there.
Warm and cold (jit lower+compile) timings of the coil-field kernels on a helias-like set of 50 FourierCoils with
96 modes. Not collected by pytest. Run: PYTHONPATH=src python tests/benchmarks/bench_coil_fields.py [--n-coils 50]
Works on both the old (coil-fields) and new (fast-eval) code: kernels that do not exist are skipped / use fallbacks."""
import argparse, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax_sbgeom.coils import FourierCoil, CoilSet, FiniteSizeCoilSet, CentroidFrame, filament_field, self_field
from jax_sbgeom.coils import fields

parser = argparse.ArgumentParser(); parser.add_argument("--n-coils", type=int, default=50); parser.add_argument("--modes", type=int, default=96)
parser.add_argument("--select", type=int, default=10)
args = parser.parse_args()
NC, NM, NSEL = args.n_coils, args.modes, args.select

# coil k: a wavy loop of radius ~1.5 m around a torus of major radius 5 m, rotated by 2 pi k / NC about z
k = jnp.arange(1, NM + 1)
def make(phi):
    e1 = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0]); ez = jnp.array([0.0, 0.0, 1.0])
    cos = jnp.zeros((NM, 3)).at[0].set(1.5 * e1).at[1].set(0.1 * ez).at[1].add(0.05 * e1)
    sin = jnp.zeros((NM, 3)).at[0].set(1.5 * ez) + 0.05 * jnp.sin(k)[:, None] / k[:, None] ** 2 * e1
    return FourierCoil(fourier_cos=cos, fourier_sin=sin, centre_i=5.0 * e1)
cset = CoilSet.from_list([make(2 * jnp.pi * i / NC) for i in range(NC)])
fcset = FiniteSizeCoilSet.from_coilset(cset, CentroidFrame)
coils = cset.coils
currents = jnp.full(NC, 1e6)
idx = jnp.arange(NSEL)
s256 = jnp.arange(256) / 256
width, height = 0.5, 0.4


def block(x):
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), x)


def bench(name, f, *a, reps=50):
    jf = jax.jit(f)
    t0 = time.perf_counter(); comp = jf.lower(*a).compile(); tc = time.perf_counter() - t0
    for _ in range(30):                      # warm-up (GPU clocks ramp up slowly)
        block(comp(*a))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); block(comp(*a)); ts.append(time.perf_counter() - t0)
    tw = sorted(ts)[len(ts) // 2]            # median, one synchronisation per call
    t0 = time.perf_counter()
    for _ in range(reps):
        out = comp(*a)
    block(out)
    tp = (time.perf_counter() - t0) / reps   # back-to-back dispatch, one synchronisation: the device time
    print(f"{name:36s} sync {tw * 1e3:8.2f} ms  pipelined {tp * 1e3:8.2f} ms  compile {tc:6.2f} s", flush=True)


def positions(c, s):
    return jax.vmap(lambda ci: ci.position(s))(c)


def self_one(c, s):
    return self_field(jax.tree_util.tree_map(lambda a: a[0], c), 1e6, width, height, s, u=-1.0, v=0.0, n_src=1024)

def fil(c, cur, pts):
    return filament_field(c, cur, pts, 256, 64)

pts = positions(coils, s256)[0]
print("devices:", jax.devices())
bench(f"positions {NC} coils x 256", positions, coils, s256)
bench("self_field, 1 coil (n_src 1024)", self_one, fcset.coils, s256)
bench(f"filament_field {NC} coils -> 256 pts", fil, cset, currents, pts)
