from symjit import compile_func
from sympy import *
import numpy as np

points = []
for p in range(0, 300, 20):
    points.extend([10.0**p, -10.0**p, 10.0**-p, -10.0**-p, 10.0**p*1j, -10.0**p*1j, 10.0**-p*1j, -10.0**-p*1j])

x, y = symbols("x y")

def test_division():
    g0 = compile_func([x, y], [x / y], dtype="complex128", stable_math=False)
    g1 = compile_func([x, y], [x / y], dtype="complex128", stable_math=True, fast_complex=False)
    g2 = compile_func([x, y], [x / y], dtype="complex128", stable_math=True, fast_complex=True)

    for den in points:
        for num in points:
            b0 = g0(num, den)[0]
            b1 = g1(num, den)[0]
            b2 = g2(num, den)[0]
            r = (num + 0j) / (den + 0j)
            print(f"{num}/{den}: {r}\t{b1}\t{b2} vs {b0}")


def test_sqrt():
    g0 = compile_func([x], [sqrt(x)], dtype="complex128", stable_math=False)
    g1 = compile_func([x], [sqrt(x)], dtype="complex128", stable_math=True, fast_complex=False)
    g2 = compile_func([x], [sqrt(x)], dtype="complex128", stable_math=True, fast_complex=True)

    for p in points:
        b0 = g0(p)[0]
        b1 = g1(p)[0]
        b2 = g2(p)[0]
        r = np.sqrt(p)
        print(f"{p}: {r}\t{b1}\t{b2} vs {b0}")


test_division()
test_sqrt()
