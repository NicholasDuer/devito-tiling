#!/bin/bash
csv_name='results.csv'

num_iterations=3
space_order=8

echo -n >$csv_name
for num_ranks in 2 4
do
    for time in 200
    do
        for x in 300 400
        do
            for y in 200 300
            do
                for z in 200
                do
                    for iteration in `seq 1 $num_iterations`
                    do
                    echo -n "$num_ranks,$time,$x,$y,$z,$iteration" >> $csv_name
                    DEVITO_AUTOTUNING=aggressive OMP_NUM_THREADS=8 DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n $num_ranks numactl --membind=0,1 --cpunodebind=0,1 python3 mpi_experiment.py -d $x $y $z --nt $time -so $space_order
                    echo -en "\n" >> $csv_name
                    done
                done
            done
        done
    done
done
