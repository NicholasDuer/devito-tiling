from matplotlib.pyplot import pause # noqa
from devito import configuration
import matplotlib.pyplot as plt
import numpy as np

from devito import (Grid, Eq, TimeFunction, Operator, norm,
                    Constant, solve)
from devito.ir import Iteration, FindNodes
import argparse

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(64, 64, 64), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=2,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=1,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=40,
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
init_value = 5

# Field initialization
grid = Grid(shape=(nx, ny, nz))
u = TimeFunction(name='u', grid=grid, space_order=so, time_order=to)
u.data[:, :, :, :] = 0
u.data[0, 10, 5, 5] = init_value
u.data[0, 10, 25, 20] = -init_value

# Create an equation with second-order derivatives
a = Constant(name='a')
a = 0.5
eq = Eq(u.dt, a*u.laplace)
stencil = solve(eq, u.forward)
eq0 = Eq(u.forward, stencil)

# ======= standard implementation for image
op0 = Operator(eq0, opt=('advanced', {'mpi': False}))
op0.apply(time_M=nt, dt=dt)
uref = u.data

fig = plt.figure(figsize=(8,8))
plt.xticks([])
plt.yticks([])
plt.imshow(uref[0, 10, :, :], cmap="coolwarm_r"); 
plt.savefig("./images/poles_nt" + str(nt) + ".png")
