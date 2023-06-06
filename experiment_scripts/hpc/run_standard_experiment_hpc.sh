#!/bin/bash
num_iterations=3
num_ranks=2

devito_path="$HOME/devito"
experiment_path="$HOME/devito-tiling"
original_branch="experiment-unmodified"
modified_branch="mpi-overlapped-tiling"

csv_name_temp_results="${experiment_path}/results.csv"
csv_name_standard_mpi="${experiment_path}/laplace_standard_mpi.csv"

check_norms_script="${experiment_path}/experiment_scripts/check_norms.py"
laplace_script="${experiment_path}/experiment_scripts/mpi_experiment_laplace.py"
wave_script="${experiment_path}/experiment_scripts/mpi_experiment_laplace.py"

devito_env_path="$HOME/devito-env/bin/activate"

space_orders=(2 4 8)
experiment_dims=(256,512,512,512)

threads_per_core=10

source $devito_env_path
module load intel-suite/2020.2
module load mpi/intel-2019
module load tools/prod
module load iimpi/2021b

cd $devito_path
git checkout $original_branch
cd $experiment_path

set -e

rm -f $csv_name_temp_results
echo "num_ranks,space_order,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_standard_mpi
for space_order in space_orders
do
    for experiment_dim in ${experiment_dims[@]}
    do
        IFS=',' read -a dims <<< "${experiment_dim}"
        time=${dims[0]}
        x=${dims[1]}
        y=${dims[2]}
        z=${dims[3]}
        for iteration in `seq 1 $num_iterations`
        do
            DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=0 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
            cat $csv_name_temp_results >> $csv_name_standard_mpi
            echo -en "\n" >> $csv_name_standard_mpi

            rm $csv_name_temp_results
        done
    done
done