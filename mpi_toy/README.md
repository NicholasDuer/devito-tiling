# Reproducing MPI correctness tests
To verify our MPI implementation, we check the norm produced by a run of Devito. It should match the norm produced by the standard Devito implementation. Below we show how to test the 3D implementation on 4 ranks using a simple toy example.  

## Steps
* If not done already, clone the [Devito repository](https://github.com/devitocodes/devito)
* Activate a Python environment with the necessary dependencies:
    * [requirements.txt](https://github.com/devitocodes/devito/blob/master/requirements.txt), [requirements-optional.txt](https://github.com/devitocodes/devito/blob/master/requirements-optional.txt), [requirements-mpi.txt](https://github.com/devitocodes/devito/blob/master/requirements-mpi.txt)
* Now navigate to the root of this (devito-tiling) repository and run `DEVITO_LOGGING=DEBUG DEVITO_MPI=1 mpirun -n 4 python3 ./mpi/toy_3D.py`
* Note the hash of the cached kernel produced by Devito
* Replace the kernel with our MPI implementation using the command: `cat ./mpi/copy_mpi_3d.c > /tmp/devito-jitcache/(whatever the hash of the cached kernel is)`
* Run the new implementation using: `DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n 4 python3 ./mpi/toy_3D.py`