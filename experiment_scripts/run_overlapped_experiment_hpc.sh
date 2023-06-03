#!/bin/bash
num_iterations=4
num_ranks=4

devito_path="$HOME/devito"
experiment_path="$HOME/devito-tiling"
original_branch="experiment-unmodified"
modified_branch="mpi-overlapped-tiling"

csv_name_temp_results="${experiment_path}/results.csv"
csv_name_standard_mpi="${experiment_path}/results_standard_mpi.csv"
csv_name_overlapped="${experiment_path}/results_overlapped_mpi.csv"

norm_temp_text="norms.txt"

check_norms_script="${experiment_path}/experiment_scripts/check_norms.py"
experiment_script="${experiment_path}/experiment_scripts/mpi_experiment.py"
devito_env_path="$HOME/devito-env/bin/activate"

space_orders=(2 4 8)
time_tile_sizes=(4 8 16 32)
wavefront_dims=(32,64,96 32,64,128 64,128,256)
experiment_dims=(256,512,512,512)

threads_per_core=10

source $devito_env_path
module load intel-suite/2020.2
module load mpi/intel-2019
module load tools/prod
module load iimpi/2021b

set -e

rm -f $norm_temp_text
rm -f $csv_name_temp_results
echo "num_ranks,space_order,time_tile_size,wf_x_width,wf_y_width,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_overlapped
echo "num_ranks,space_order,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_standard_mpi
for space_order_index in `seq 0 2`
do
    space_order=${space_orders[$space_order_index]}
    for time_tile_size in ${time_tile_sizes[@]}
    do
        wf_dims_tts=${wavefront_dims[$space_order_index]}
        IFS=',' read -a wf_dims <<< "${wf_dims_tts}"
        for wf_x_index in `seq 0 2`
        do
            for wf_y_index in `seq 0 2`
            do
                if [ `expr $wf_x_index - $wf_y_index` != 2 ] && [ `expr $wf_x_index - $wf_y_index` != -2 ]
                then
                    wf_x_width=${wf_dims[$wf_x_index]}
                    wf_y_width=${wf_dims[$wf_y_index]}
                    for experiment_dim in ${experiment_dims[@]}
                    do
                        IFS=',' read -a dims <<< "${experiment_dim}"
                        time=${dims[0]}
                        x=${dims[1]}
                        y=${dims[2]}
                        z=${dims[3]}
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

                            if [ $time_tile_size -eq ${time_tile_sizes[0]} ] && [ $wf_x_index -eq 0 ] && [ $wf_y_index -eq 0 ]
                            then
                                echo -n "$num_ranks,$space_order,$time,$x,$y,$z,$iteration" >> $csv_name_standard_mpi
                                standard_autotuning="aggressive"
                            else
                                standard_autotuning="off"
                            fi
                            DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=$standard_autotuning OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=0 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                            if [ $time_tile_size -eq ${time_tile_sizes[0]} ] && [ $wf_x_index -eq 0 ] && [ $wf_y_index -eq 0 ]
                            then
                                cat $csv_name_temp_results >> $csv_name_standard_mpi
                                echo -en "\n" >> $csv_name_standard_mpi
                            fi

                            rm $csv_name_temp_results

                            python3 $check_norms_script
                            rm $norm_temp_text
                        done
                    done  
                fi
            done
        done 
    done
done
