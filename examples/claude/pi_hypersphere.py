"""Pi from the volume of the four-dimensional unit ball, with a recursive complex `Composer` kernel.

The unit ball in four dimensions (bounded by the 3-sphere) has volume pi^2 / 2; its part in the
positive orthant (all coordinates >= 0) has volume pi^2 / 32, so pi = sqrt(32 V).

A point (x1, x2, x3, x4) is a pair of complex numbers (x1 + i x2, x3 + i x4). The kernel takes an
axis-aligned box in the positive orthant as two opposite corners, `lo` and `hi` (four complex
arguments), counts which of its 16 corners lie in the unit ball (|x|^2 <= 1), and returns

- the volume of the box, if all 16 corners are inside (the ball is convex);
- 0, if none is: in the positive orthant the ball is closed toward the origin, so if any point of
  the box were inside, so would be `lo`, the corner nearest the origin (no axis-crossing test);
- otherwise, if the side is at most `2^-depth`, the volume times the fraction of the corners
  inside; else the sum of the 16 boxes obtained by halving every side (16 recursive calls).

All the non-recursive cases are `volume * n / 16` (n = corners inside), so the kernel computes that
and only branches on whether to recurse. The two-dimensional version (`rust tests/pi_area.rs` in
v227, starting from the whole square) would start here from (0,0), (0,1), (1,1), (1,0) and multiply
by 4.

The kernel's result is checked against the same algorithm in numpy (breadth-first over the boxes
of each level). All volumes are dyadic (sides 2^-k), so both sums are exact and must agree bit for
bit.

    python examples/claude/pi_hypersphere.py              # depths 1..6
    python examples/claude/pi_hypersphere.py --depth 7    # depths 1..7 (a few seconds)
"""

import argparse
import itertools
import math
import time

import numpy as np

from symjit import Composer, compile_composer

DIM = 4
CORNERS = list(itertools.product((0, 1), repeat=DIM))  # 0: lo, 1: hi, per axis


def hypersphere_orthant(min_side: float) -> Composer:
    cp = Composer(4, 1)  # lo = (arg 0, arg 1), hi = (arg 2, arg 3)

    zero, one, half = cp.constant(0.0), cp.constant(1.0), cp.constant(0.5)
    count_all = cp.constant(float(len(CORNERS)))
    fraction = cp.constant(1.0 / len(CORNERS))

    def coords(z1, z2):
        return [cp.real(z1), cp.imag(z1), cp.real(z2), cp.imag(z2)]

    lo = coords(cp.arg(0), cp.arg(1))
    hi = coords(cp.arg(2), cp.arg(3))

    # comparisons return bit masks; `and` with 1.0 turns them into 1.0 or 0.0 for counting
    def flag(mask):
        return cp.and_(mask, one)

    # n = number of corners inside the unit ball
    sq = [[cp.square(lo[k]), cp.square(hi[k])] for k in range(DIM)]
    n = None
    for corner in CORNERS:
        r2 = sq[0][corner[0]]
        for k in range(1, DIM):
            r2 = cp.fadd(r2, sq[k][corner[k]])
        inside = flag(cp.leq(r2, one))
        n = inside if n is None else cp.fadd(n, inside)

    # the boxes are cubes (halved from the unit cube), so one side is enough
    side = cp.fsub(hi[0], lo[0])
    volume = cp.fmul(cp.fmul(side, side), cp.fmul(side, side))
    leaf = cp.fmul(volume, cp.fmul(n, fraction))

    recurse = cp.and_(
        cp.and_(cp.neq(n, zero), cp.neq(n, count_all)),
        cp.gt(side, cp.constant(min_side)),
    )

    # the recursive step: 16 sub-boxes, `call(None, ...)` calls the kernel itself
    rec, base = cp.new_block(), cp.new_block()
    mid = [rec.fmul(rec.fadd(lo[k], hi[k]), half) for k in range(DIM)]
    bounds = [[lo[k], mid[k], hi[k]] for k in range(DIM)]

    total = None
    for corner in CORNERS:
        a = [bounds[k][corner[k]] for k in range(DIM)]
        b = [bounds[k][corner[k] + 1] for k in range(DIM)]
        part = rec.call(
            None,
            rec.complex(a[0], a[1]),
            rec.complex(a[2], a[3]),
            rec.complex(b[0], b[1]),
            rec.complex(b[2], b[3]),
        )
        total = part if total is None else rec.fadd(total, part)

    cp.append_if_else(recurse, rec, base)
    cp.assign(cp.out(0), cp.join(recurse, total, leaf))
    return cp


def reference(min_side: float) -> float:
    """The same algorithm in numpy, level by level over the boxes still to be split."""
    lo = np.zeros((1, DIM))
    side = 1.0
    corners = np.array(CORNERS, dtype=float)
    total = 0.0

    while len(lo):
        # (boxes, corners): squared radius of every corner of every box
        pts = lo[:, None, :] + side * corners[None, :, :]
        n = (np.sum(pts * pts, axis=2) <= 1.0).sum(axis=1)
        volume = side**DIM
        recurse = (n != 0) & (n != len(CORNERS)) & (side > min_side)
        total += float(np.sum(volume * n[~recurse] / len(CORNERS)))
        side *= 0.5
        lo = (lo[recurse][:, None, :] + side * corners[None, :, :]).reshape(-1, DIM)

    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--depth", type=int, default=6, help="smallest side is 2^-depth")
    args = parser.parse_args()

    print(f"{'depth':>5} {'pi estimate':>18} {'error':>10} {'compile':>9} {'kernel':>9} {'numpy':>9}")

    for depth in range(1, args.depth + 1):
        min_side = 2.0**-depth

        t0 = time.perf_counter()
        f = compile_composer(hypersphere_orthant(min_side), dtype="complex128")
        t1 = time.perf_counter()
        v = f.evaluate_complex([[0j, 0j, 1 + 1j, 1 + 1j]])[0, 0]
        t2 = time.perf_counter()
        ref = reference(min_side)
        t3 = time.perf_counter()

        assert v.imag == 0.0, v
        assert v.real == ref, (depth, v.real, ref)

        pi = math.sqrt(32.0 * v.real)
        print(
            f"{depth:>5} {pi:>18.12f} {pi - math.pi:>10.2e} "
            f"{1e3 * (t1 - t0):>7.1f}ms {1e3 * (t2 - t1):>7.1f}ms {1e3 * (t3 - t2):>7.1f}ms"
        )

    print("ok: the kernel matches the numpy reference at every depth")


if __name__ == "__main__":
    main()
