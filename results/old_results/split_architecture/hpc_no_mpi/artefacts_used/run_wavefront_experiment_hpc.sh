#!/bin/bash
num_iterations=2

devito_path="$HOME/devito"
experiment_path="$HOME/devito-tiling"
original_branch="experiment-unmodified"

csv_name_temp_results="${experiment_path}/results.csv"
csv_name_wavefront="${experiment_path}/results_wavefront_mpi.csv"
csv_name_standard="${experiment_path}/results_standard_openmp.csv"

norm_temp_text="norms.txt"

check_norms_script="${experiment_path}/experiment_scripts/check_norms.py"
experiment_script="${experiment_path}/experiment_scripts/wavefront_experiment.py"
devito_env_path="$HOME/devito-env/bin/activate"

space_orders=(2 4 8)
time_tile_sizes=(4 8 16 32)
wavefront_dims=(32,64,96 32,64,128 64,128,256)
experiment_dims=(256,256,256,256 256,512,256,256 256,256,512,256 256,512,512,512 512,256,256,256)

threads_per_core=10

module load intel-suite/2020.2
module load tools/prod

source $devito_env_path
cd $devito_path
git checkout "${original_branch}"
cd $experiment_path

set -e

rm -f $norm_temp_text
rm -f $csv_name_temp_results
echo "space_order,time_tile_size,wf_x_width,wf_y_width,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss" >$csv_name_wavefront
echo "space_order,time,x_size,y_size,z_size,repeat_num,elapsed_time,oi,gflopss,gpointss" >$csv_name_standard
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
                            echo -n "$space_order,$time_tile_size,$wf_x_width,$wf_y_width,$time,$x,$y,$z,$iteration" >> $csv_name_wavefront
                            WF_HEIGHT=$time_tile_size WF_X_WIDTH=$wf_x_width WF_Y_WIDTH=$wf_y_width DEVITO_AUTOTUNING=aggressive OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_JIT_BACKDOOR=1 python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                            cat $csv_name_temp_results >> $csv_name_wavefront
                            echo -en "\n" >> $csv_name_wavefront
                            rm $csv_name_temp_results

                            if [ $time_tile_size -eq ${time_tile_sizes[0]} ] && [ $wf_x_index -eq 0 ] && [ $wf_y_index -eq 0 ]
                            then
                                echo -n "$space_order,$time,$x,$y,$z,$iteration" >> $csv_name_standard
                                standard_autotuning="aggressive"
                            else
                                standard_autotuning="off"
                            fi    
                            DEVITO_PROFILING=advanced2 DEVITO_AUTOTUNING=$standard_autotuning OMP_PROC_BIND=close OMP_NUM_THREADS=$threads_per_core OMP_PLACES=cores DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG DEVITO_JIT_BACKDOOR=0 python3 $experiment_script -d $x $y $z --nt $time -so $space_order
                            if [ $time_tile_size -eq ${time_tile_sizes[0]} ] && [ $wf_x_index -eq 0 ] && [ $wf_y_index -eq 0 ]
                            then
                                cat $csv_name_temp_results >> $csv_name_standard
                                echo -en "\n" >> $csv_name_standard
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
