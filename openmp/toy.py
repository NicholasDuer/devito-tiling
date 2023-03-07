import pytest
import numpy as np

from devito import (Grid, Eq, Function, TimeFunction, Operator, norm,
                    Constant, solve)
from devito.ir import Expression, Iteration, FindNodes, FindSymbols
import argparse

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(32, 32, 32), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=4,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=1,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=32,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=2, type=int, nargs="+",
                    help="Block levels")
args = parser.parse_args()


nx, ny, nz = args.shape
nt = args.nt
nu = .5
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
dz = 2. / (nz - 1)
sigma = .25
dt = sigma * dx * dz * dy / nu


so = args.space_order
to = 1

# Initialise u
init_value = 6.5

# Field initialization
grid = Grid(shape=(nx, ny, nz))
u = TimeFunction(name='u', grid=grid, space_order=so, time_order=to)
u.data[:, :, :] = init_value

# Create an equation with second-order derivatives
a = Constant(name='a')
eq = Eq(u.dt, a*u.laplace + 0.1, subdomain=grid.interior)
stencil = solve(eq, u.forward)
eq0 = Eq(u.forward, stencil)

# List comprehension would need explicit locals/globals mappings to eval
op0 = Operator(eq0, opt=('advanced'))
op0.apply(time_M=nt, dt=dt)

norm_u = norm(u)
print(norm_u)


u.data[:] = init_value
assert(np.isclose(norm_u, 2353.541, atol=1e-3, rtol=0))
# norm for t = 100, 256x200x300 36026.656
# norm for t = 40, 10x10x10 291.68246
# norm for t = 128, 512x512x512 106496.0
# norm for t = 64, 128x128x128 13312.01
# norm for t = 64, 128x256x128 18826.045
# norm for t = 32, 32x32x32 1664.1122