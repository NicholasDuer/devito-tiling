import pytest
import numpy as np

from devito import (Grid, Eq, Function, TimeFunction, Operator, norm,
                    Constant, solve, configuration)
from devito.ir import Expression, Iteration, FindNodes, FindSymbols
import argparse
from mpi4py import MPI
import ipyparallel as ipp

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(10), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-nt", "--nt", default=8,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=2, type=int, nargs="+",
                    help="Block levels")
args = parser.parse_args()


nx = args.shape
nt = args.nt

init_value = 0

# Field initialization
grid = Grid(shape=(nx))
u = TimeFunction(name='u', grid=grid)
u.data[:, :] = init_value

# Create an equation with second-order derivatives
a = Constant(name='a')

# List comprehension would need explicit locals/globals mappings to eval
op0 = Operator(Eq(u.forward, u.dx + 1))
op0.apply(time_M=nt)

norm_u = norm(u)
print(norm_u)
assert(np.isclose(norm_u, 2392802800.0))
# x = 10, t = 8, 2392802800.0
# x = 20, t = 8, 969778000000.0
# x = 10, t = 16, 2.10698e+19
