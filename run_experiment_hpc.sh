#!/bin/bash
num_iterations=2
num_ranks=4

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
wavefront_dims=(32,64,96 64,128,196 128,196,256 128,196,256)
experiment_dims=(256,256,256,256 256,512,256,256 256,256,512,256 256,512,512,512 512,256,256,256)

threads_per_core=10

source $devito_env_path
module load intel-suite/2020.2
module load mpi/intel-2019

set -e

rm -f $norm_temp_text
rm -f $csv_name_temp_results
echo "num_ranks,space_order,time_tile_size,wf_x_width,wf_y_width,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_overlapped
echo "num_ranks,space_order,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss,haloupdate0" >$csv_name_standard_mpi
for space_order in ${space_orders[@]}
do
    for tts_index in `seq 0 3`
    do
        time_tile_size=${time_tile_sizes[$tts_index]}
        wf_dims_tts=${wavefront_dims[$tts_index]}
        IFS=',' read -a wf_dims <<< "{$wf_dims_tts}"
        for wf_x_index in `seq 0 2`
        do
<<<<<<< HEAD
            for wf_y_index in `seq 0 2`
            do
                if [ `expr $wf_x_index - $wf_y_index` != 2 ] && [ `expr $wf_x_index - $wf_y_index` != -2 ]
                then
                    wf_x_width=${wf_dims_tts[$wf_x_index]}
                    wf_y_width=${wf_dims_tts[$wf_y_index]}
                    for experiment_dim in ${experiment_dims[@]}
=======
            for wf_y_width in ${wf_y_widths[@]}
            do  
                for time in ${t_vals[@]}
                do
                    for x in ${x_vals[@]}
>>>>>>> c2a30ddf5e4a3d460ab73ad0d2b74f2e6f9ddcd8
                    do
                        IFS=',' read -a dims <<< "${experiment_dim}"
                        time=${dims[0]}
                        x_size=${dims[1]}
                        y_size=${dims[2]}
                        z_size=${dims[3]}
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

<<<<<<< HEAD
                            if [ $time_tile_size -eq ${time_tile_sizes[0]} ] && [ $wf_x_width -eq ${wf_x_widths[0]} ] && [ $wf_y_width -eq ${wf_x_widths[0]} ]
                            then
                                    echo -n "$num_ranks,$space_order,$time,$x,$y,$z,$iteration" >> $csv_name_standard_mpi
                            fi
                            DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=0 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                            if [ $time_tile_size -eq ${time_tile_sizes[0]} ] && [ $wf_x_width -eq ${wf_x_widths[0]} ] && [ $wf_y_width -eq ${wf_x_widths[0]} ]
                            then
                                cat $csv_name_temp_results >> $csv_name_standard_mpi
                                echo -en "\n" >> $csv_name_standard_mpi
                            fi
=======
                                if [ $time_tile_size -eq ${time_tile_sizes[0]} ] && [ $wf_x_width -eq ${wf_x_widths[0]} ] && [ $wf_y_width -eq ${wf_x_widths[0]} ]
                                then
                                    echo -n "$num_ranks,$space_order,$time,$x,$y,$z,$iteration" >> $csv_name_standard_mpi
                                fi
                                DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_MPI=1 DEVITO_JIT_BACKDOOR=0 mpirun -n $num_ranks --bind-to socket --map-by socket python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                                if [ $time_tile_size -eq ${time_tile_sizes[0]} ] && [ $wf_x_width -eq ${wf_x_widths[0]} ] && [ $wf_y_width -eq ${wf_x_widths[0]} ]
                                then
                                    cat $csv_name_temp_results >> $csv_name_standard_mpi
                                    echo -en "\n" >> $csv_name_standard_mpi
                                fi
>>>>>>> c2a30ddf5e4a3d460ab73ad0d2b74f2e6f9ddcd8

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
