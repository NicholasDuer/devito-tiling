# Reproducing MPI correctness tests
To verify our MPI implementation, we check the norm produced by a run of Devito. It should match the norm produced by the standard Devito implementation. Below we show how to test the 3D implementation on 4 ranks on the Laplace equation on a 128x128x128 grid.

The MPI implementation for the 3D Laplace equation can be verified for the following space orders: 2, 4, 8.
## Steps
* Clone the forked [Devito repository](https://github.com/NicholasDuer/devito)
* Navigate into the Devito repository and swap to the [mpi-overlapped-tiling](https://github.com/NicholasDuer/devito/tree/mpi-overlapped-tiling) branch
* Activate a Python environment and install the necessary dependencies:
    * [requirements.txt](https://github.com/devitocodes/devito/blob/master/requirements.txt), [requirements-optional.txt](https://github.com/devitocodes/devito/blob/master/requirements-optional.txt), [requirements-mpi.txt](https://github.com/devitocodes/devito/blob/master/requirements-mpi.txt)
* `cd` out of the Devito repository and clone the [devito-tiling repository](https://github.com/NicholasDuer/devito-tiling). Navigate to the root of the repository
* Run the command, setting the desired space order: `DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=0 mpirun -n 1 python3 mpi_test.py -d 128 128 128 --nt 64 -so <desired space order here>`
* Note the file path of the **second** cached kernel produced by Devito: 
    * There will be two lines in the output containing the content: *Operator \`Kernel` fetched \<file path>*
    * Find the **second** appearance of this line and note the file path
    * It will look something like: */tmp/devito-jitcache-uid1000/\<a long hash code>.c*
* Replace the kernel with our MPI implementation using the command: `cat mpi_laplace_<desired space order here>so.c > <file path from previous step>`
* Run the new implementation using: `DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n 4 python3 mpi_test.py -d 128 128 128 --nt 64 -so <desired space order here>`