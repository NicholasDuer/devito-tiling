import pytest
import numpy as np

from devito import (Grid, Eq, Function, TimeFunction, Operator, norm,
                    Constant, solve)
from devito.ir import Expression, Iteration, FindNodes, FindSymbols
import argparse

grid = Grid(shape=(10, 10))
u = TimeFunction(name='f', grid=grid)

# Some variable declarations
nx, nt = 10, 10
dx = 2. / (nx - 1)
sigma = .2
dt = sigma * dx

grid = Grid(shape=(nx, ), extent=(1.,))
u = TimeFunction(name='u', grid=grid, space_order=2)

eq = Eq(u.dt, u.dx2, grid=grid)
stencil = solve(eq, u.forward)

# Boundary conditions
t = grid.stepping_dim
bc_left = Eq(u[t + 1, 0], 0.)
bc_right = Eq(u[t + 1, nx-1], 0.)

op = Operator([Eq(u.forward, stencil), bc_left, bc_right])
op.apply(time_M=nt, dt=dt)
# norm for t = 100, 256x200x300 36026.656
# norm for t = 40, 10x10x10 291.68246
# norm for t = 128, 512x512x512 106496.0
# norm for t = 64, 128x128x128 13312.01
# norm for t = 64, 128x256x128 18826.045
# norm for t = 32, 32x32x32 1664.1122