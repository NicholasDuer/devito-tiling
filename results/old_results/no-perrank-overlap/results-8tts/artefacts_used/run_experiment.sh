#!/bin/bash
num_iterations=3
num_ranks=2
space_order=8

devito_path="$HOME/devito"
experiment_path="$HOME/devito-tiling"
original_branch="experiment-unmodified"
modified_branch="mpi-overlapped-tiling"

csv_name_temp_results="results.csv"
csv_name_openmp="results_openmp_${space_order}so.csv"
csv_name_standard_mpi="results_standard_mpi_${space_order}so.csv"
csv_name_overlapped="results_overlapped_mpi_${space_order}so.csv"

test_norms_script="mpi_test_norms.py"
experiment_script="mpi_experiment.py"
devito_env_path="../devito-env/bin/activate"

t_vals=(200)
x_vals=(300 400)
y_vals=(200 300)
z_vals=(200)

threads_per_core=8

source $devito_env_path

set -e
cd $devito_path
git checkout $modified_branch
cd $experiment_path


for time in ${t_vals[@]}
do
    for x in ${x_vals[@]}
    do
        for y in ${y_vals[@]}
        do
            for z in ${z_vals[@]}
            do
	    DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $test_norms_script -d $x $y $z --nt $time -so $space_order
            done
        done
    done
done

cd $devito_path
git checkout $original_branch
cd $experiment_path

echo "num_ranks,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss" >$csv_name_temp_results
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
                echo -n "$num_ranks,$time,$x,$y,$z,$iteration" >> $csv_name_temp_results
                DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_JIT_BACKDOOR=0 python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                echo -en "\n" >> $csv_name_temp_results
                done
            done
        done
    done
done

cat $csv_name_temp_results > $csv_name_openmp

cd $devito_path
git checkout $modified_branch
cd $experiment_path

echo "num_ranks,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_temp_results
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
                echo -n "$num_ranks,$time,$x,$y,$z,$iteration" >> $csv_name_temp_results
                DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                echo -en "\n" >> $csv_name_temp_results
                done
            done
        done
    done
done

cat $csv_name_temp_results > $csv_name_overlapped

cd $devito_path
git checkout $original_branch
cd $experiment_path

echo "num_ranks,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_temp_results
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
                echo -n "$num_ranks,$time,$x,$y,$z,$iteration" >> $csv_name_temp_results
                DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=0 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                echo -en "\n" >> $csv_name_temp_results
                done
            done
        done
    done
done

cat $csv_name_temp_results > $csv_name_standard_mpi
rm $csv_name_temp_results
