# ruff: noqa: E401, E501, I001, E702
import os, sys, time
os.environ.update({name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
import numpy as np
n, repeats = (int(sys.argv[1]) if len(sys.argv) > 1 else 500_000), 20
rng, alpha, h, eta = np.random.default_rng(20260822), 1.5, 0.075, 0.5
grid = np.linspace(0.0, np.log1p(1e5), 4000)
table, x = -np.expm1(grid)/(1.0+np.expm1(grid)), rng.standard_normal(n)
decay, mult, noise = np.exp(-eta*h/alpha), -np.expm1(-eta*h/alpha)/(eta/alpha), (-np.expm1(-eta*h))**(1/alpha)
def cms():
    u = np.pi*(rng.random(n)-0.5)
    w = -np.log(rng.random(n))
    return np.sin(alpha*u)/np.cos(u)**(1/alpha)*(np.cos((1-alpha)*u)/w)**((1-alpha)/alpha)
def score():
    r = np.abs(x)
    q = np.log1p(np.minimum(r, 1e5))/grid[-1]*(grid.size-1)
    i = np.minimum(q.astype(np.int64), grid.size-2); f = q-i
    inside = np.sign(x)*(table[i]+f*(table[i+1]-table[i]))
    return np.where(r <= 1e5, inside, -2.0*np.sign(x)/np.maximum(r, 1e-300))
def step():
    s = score()
    return decay*x + mult*(x+alpha*s) + noise*cms()
def rate(fn):
    fn(); start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    return repeats*n/(time.perf_counter()-start), float(result[0])
print(sys.version.split()[0], f"NumPy={np.__version__}", f"n={n}", f"repeats={repeats}")
for label, fn in (("CMS draws/s", cms), ("score eval/s", score), ("EI particle-steps/s", step)):
    print(f"{label:22s} {rate(fn)[0]:.6g}")
