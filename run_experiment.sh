#!/bin/bash
num_iterations=3
num_ranks=2
space_order=2

csv_name_temp_results="results.csv"
csv_name_openmp="results_openmp_${space_order}so.csv"
csv_name_standard_mpi="results_standard_mpi_${space_order}so.csv"
csv_name_overlapped="results_overlapped_mpi_${space_order}so.csv"

test_norms_script="mpi_test_norms.py"
experiment_script="mpi_experiment.py"

t_vals=(200)
x_vals=(300 400)
y_vals=(200 300)
z_vals=(200)

threads_per_core=8

set -e

for time in ${t_vals[@]}
do
    for x in ${x_vals[@]}
    do
        for y in ${y_vals[@]}
        do
            for z in ${z_vals[@]}
            do
            DEVITO_AUTOTUNING=aggressive OMP_NUM_THREADS=$threads_per_core DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n $num_ranks numactl --membind=0,1 --cpunodebind=0,1 python3 $test_norms_script -d $x $y $z --nt $time -so $space_order           
            done
        done
    done
done

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
                DEVITO_AUTOTUNING=aggressive OMP_NUM_THREADS=$threads_per_core DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_JIT_BACKDOOR=0 python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                echo -en "\n" >> $csv_name_temp_results
                done
            done
        done
    done
done

cat $csv_name_temp_results > $csv_name_openmp

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
                DEVITO_AUTOTUNING=aggressive OMP_NUM_THREADS=$threads_per_core DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=0 mpirun -n $num_ranks numactl --membind=0,1 --cpunodebind=0,1 python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                echo -en "\n" >> $csv_name_temp_results
                done
            done
        done
    done
done

cat $csv_name_temp_results > $csv_name_standard_mpi

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
                DEVITO_AUTOTUNING=aggressive OMP_NUM_THREADS=$threads_per_core DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n $num_ranks numactl --membind=0,1 --cpunodebind=0,1 python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                echo -en "\n" >> $csv_name_temp_results
                done
            done
        done
    done
done

cat $csv_name_temp_results > $csv_name_overlapped
rm $csv_name_temp_results