#!/bin/bash
csv_name='results.csv'
min_size_exp=7
max_size_exp=10

min_time_exp=7
max_time_exp=10

num_iterations=3
space_order=2

echo -n >$csv_name
for num_ranks in 2 4
do
    for time_exp in `seq $min_time_exp $max_time_exp`
    do
        for x_exp in `seq $min_size_exp $max_size_exp`
        do
            for y_exp in `seq $min_size_exp $max_size_exp`
            do
                for z_exp in `seq $min_size_exp $max_size_exp`
                do
                    for iteration in `seq 1 $num_iterations`
                    do
                    time=$((2**$time_exp))
                    x=$((2**$x_exp))
                    y=$((2**$y_exp))
                    z=$((2**$z_exp))
                    echo -n "$num_ranks,$time,$x,$y,$z,$iteration" >> $csv_name
                    DEVITO_AUTOTUNING=aggressive OMP_NUM_THREADS=8 DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n $num_ranks numactl --membind=0,1 --cpunodebind=0,1 python3 mpi_experiment.py -d $x $y $z --nt $time -so $space_order
                    echo -en "\n" >> $csv_name
                    done
                done
            done
        done
    done
done