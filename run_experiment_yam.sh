#!/bin/bash
num_iterations=3
num_ranks=2

devito_path="$HOME/devito"
experiment_path="$HOME/devito-tiling"
original_branch="experiment-unmodified"
modified_branch="mpi-overlapped-tiling"

csv_name_temp_results="results.csv"
csv_name_standard_mpi="results_standard_mpi.csv"
csv_name_overlapped="results_overlapped_mpi.csv"

norm_temp_text="norms.txt"

check_norms_script="check_norms.py"
experiment_script="mpi_experiment.py"
devito_env_path="../devito-env/bin/activate"

space_orders=(2 4 8)
time_tile_sizes=(4 8 16 32)
wf_x_widths=(16 32 64 96 128 196 256)
wf_y_widths=(16 32 64 96 128 196 256)

t_vals=(256)
x_vals=(256 512)
y_vals=(256 512)
z_vals=(256 512)

threads_per_core=8

source $devito_env_path

set -e

rm -f $norm_temp_text
rm -f $csv_name_temp_results
echo "num_ranks,space_order,time_tile_size,wf_x_width,wf_y_width,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_overlapped
echo "num_ranks,space_order,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_standard_mpi
for space_order in ${space_orders[@]}
do
    for time_tile_size in ${time_tile_sizes[@]}
    do
        for wf_x_width in ${wf_x_widths[@]}
        do
            for wf_y_width in ${wf_y_width[@]}
            do  
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
                                cd $devito_path
                                git checkout "${modified_branch}"
                                cd $experiment_path
                                echo -n "$num_ranks,$space_order,$time_tile_size,$wf_x_width,$wf_y_width,$time,$x,$y,$z,$iteration" >> $csv_name_overlapped
                                TIME_TILE_SIZE=$time_tile_size WF_X_WIDTH=$wf_x_width WF_Y_WIDTH=$wf_y_width DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=1 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                                cat $csv_name_temp_results >> $csv_name_overlapped
                                echo -en "\n" >> $csv_name_overlapped
                                rm $csv_name_temp_results

                                cd $devito_path
                                git checkout $original_branch
                                cd $experiment_path

                                if [ $time_tile_size -eq ${time_tile_sizes[0]} ] && [ $wf_x_width -eq ${wf_x_widths[0]} ] && [ $wf_y_width -eq ${wf_x_widths[0]} ]
                                then
                                        echo -n "$num_ranks,$space_order,$time,$x,$y,$z,$iteration" >> $csv_name_standard_mpi
                                fi
                                DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=0 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                                if [ $time_tile_size -eq ${time_tile_sizes[0]} ] && [ $wf_x_width -eq ${wf_x_widths[0]} ] && [ $wf_y_width -eq ${wf_x_widths[0]} ]
                                then
                                    cat $csv_name_temp_results >> $csv_name_standard_mpi
                                        secho -en "\n" >> $csv_name_standard_mpi
                                fi

                                rm $csv_name_temp_results

                                python3 $check_norms_script
                                rm $norm_temp_text
                                done
                            done
                        done
                    done
                done
            done
        done        
    done    
done
