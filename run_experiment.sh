#!/bin/bash
num_iterations=3
num_ranks=2
space_order=8

devito_path="$HOME/devito"
experiment_path="$HOME/devito-tiling"
original_branch="experiment-unmodified"
modified_branch="mpi-overlapped-tiling"

csv_name_temp_results="results.csv"
csv_name_standard_mpi="results_standard_mpi_${space_order}so.csv"
csv_name_overlapped="results_overlapped_mpi_${space_order}so.csv"

norm_temp_text="norms.txt"

check_norms_script="check_norms.py"
experiment_script="mpi_experiment.py"
devito_env_path="../devito-env/bin/activate"

t_vals=(256)
x_vals=(256 512)
y_vals=(256 512)
z_vals=(256)

threads_per_core=8

source $devito_env_path

set -e

echo "num_ranks,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_overlapped
echo "num_ranks,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_standard_mpi
for time in ${t_vals[@]}
do
    for x in ${x_vals[@]}
    do
        for y in ${y_vals[@]}
        do
            for z in ${z_vals[@]}
            do
                for iteration in `seq 1 $num_iterations`
                do
                echo "" >$norm_temp_text
                cd $devito_path
                git checkout $modified_branch
                cd $experiment_path
                echo -n "$num_ranks,$time,$x,$y,$z,$iteration" >> $csv_name_overlapped
                DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                cat $csv_name_temp_results >> $csv_name_overlapped
                echo -en "\n" >> $csv_name_standard_mpi

                cd $devito_path
                git checkout $original_branch
                cd $experiment_path
                echo -n "$num_ranks,$time,$x,$y,$z,$iteration" >> $csv_name_standard_mpi
                DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=0 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                cat $csv_name_temp_results >> $csv_name_standard_mpi
                echo -en "\n" >> $csv_name_standard_mpi

                python3 $check_norms_script
                done
            done
        done
    done
done

rm $csv_name_temp_results
rm $norm_temp_text