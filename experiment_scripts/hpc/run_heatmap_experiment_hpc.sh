#!/bin/bash
num_iterations=3
num_ranks=2

devito_path="$HOME/devito"
experiment_path="$HOME/devito-tiling"
original_branch="experiment-unmodified"
modified_branch="mpi-overlapped-tiling"

csv_name_temp_results="${experiment_path}/results.csv"
csv_name_overlapped="${experiment_path}/laplace_overlapped_mpi.csv"

experiment_script="${experiment_path}/experiment_scripts/mpi_experiment_laplace.py"
devito_env_path="$HOME/devito-env/bin/activate"

space_orders=(2 4 8)
time_tile_sizes=(4 8 16 32)
wavefront_dims=(16 32 64 96 128 196 256)
experiment_dims=(256,512,512,512)

threads_per_core=8

source $devito_env_path
cd $devito_path
git checkout "${modified_branch}"
cd $experiment_path

set -e

rm -f $norm_temp_text
rm -f $csv_name_temp_results
echo "num_ranks,space_order,time_tile_size,wf_x_width,wf_y_width,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_overlapped
for space_order in ${space_orders[@]}
do
    for time_tile_size in ${time_tile_sizes[@]}
    do
        for wf_x_width in ${wavefront_dims[@]}
        do
            for wf_y_width in ${wavefront_dims[@]}
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
                        echo -n "$num_ranks,$space_order,$time_tile_size,$wf_x_width,$wf_y_width,$time,$x,$y,$z,$iteration" >> $csv_name_overlapped
                        TIME_TILE_SIZE=$time_tile_size WF_X_WIDTH=$wf_x_width WF_Y_WIDTH=$wf_y_width DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                        cat $csv_name_temp_results >> $csv_name_overlapped
                        echo -en "\n" >> $csv_name_overlapped
                        rm $csv_name_temp_results
                    done    
                done  
            done
        done 
    done
done
