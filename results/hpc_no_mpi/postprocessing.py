from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns

results_folder = "results/"
graphs_folder = "graphs/"
heatmaps_folder = graphs_folder + "heatmaps/"
platform_name = "HPC"

space_orders = [2,4,8]
time_tile_sizes = [4,8,16,32]

wavefront_csv = pd.read_csv(results_folder + "./results_wavefront_openmp.csv")
standard_csv = pd.read_csv(results_folder + "./results_standard_openmp.csv")

# Average across repeats
wavefront_csv = wavefront_csv.groupby(['space_order', 'time_tile_size','wf_x_width','wf_y_width', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])
standard_csv = standard_csv.groupby(['space_order', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])

wavefront_csv['experiment_name'] = wavefront_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
standard_csv['experiment_name'] = standard_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
experiment_names = wavefront_csv['experiment_name'].unique()
last_experiment = experiment_names[-1]
experiment_names[-1] = experiment_names[-2]
experiment_names[-2] = last_experiment

def plot_elapsed_times_bars():
    for so in space_orders:
        wavefront_csv_so = wavefront_csv.loc[wavefront_csv['space_order'] == so]
        standard_csv_so = standard_csv.loc[standard_csv['space_order'] == so]

        def get_result_df(column_name):
            results = []
            for experiment_name in experiment_names:
                result = [experiment_name]
                for tts in time_tile_sizes:
                    result.append(wavefront_csv_so.loc[(wavefront_csv_so['time_tile_size'] == tts) & (wavefront_csv_so['experiment_name'] == experiment_name)][column_name].min())
                result.append(standard_csv_so.loc[standard_csv_so['experiment_name'] == experiment_name][column_name].min())
                results.append(result)
            results = pd.DataFrame(results)
            results.columns=['Dimensions', 'TTS=4', 'TTS=8', 'TTS=16', 'TTS=32', 'Standard MPI']
            return results 

        results = get_result_df('elapsed_time')
        results.plot(x='Dimensions', kind='bar', rot=10, ylabel="Time elapsed (s)",title="Elapsed Times, " +  platform_name + " Laplace Experiments, SO: " + str(so))
        plt.savefig(graphs_folder + "elapsed_time_" + str(so) + "so")

plot_elapsed_times_bars()
