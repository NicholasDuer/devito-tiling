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

overlapped_csv = pd.read_csv(results_folder + "./results_overlapped_mpi.csv")
overlapped_heatmap_csv = pd.read_csv(results_folder + "./results_overlapped_heatmaps_mpi.csv")
standard_csv = pd.read_csv(results_folder + "/results_standard_mpi.csv")

# Average across repeats
overlapped_csv = overlapped_csv.groupby(['num_ranks', 'space_order', 'time_tile_size','wf_x_width','wf_y_width', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])
overlapped_heatmap_csv = overlapped_heatmap_csv.groupby(['num_ranks', 'space_order', 'time_tile_size','wf_x_width','wf_y_width', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])
standard_csv = standard_csv.groupby(['num_ranks', 'space_order', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])

overlapped_csv['experiment_name'] = overlapped_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
standard_csv['experiment_name'] = standard_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
experiment_names = overlapped_csv['experiment_name'].unique()
last_experiment = experiment_names[-1]
experiment_names[-1] = experiment_names[-2]
experiment_names[-2] = last_experiment

overlapped_csv['computation_time'] = overlapped_csv['elapsed_time'] - overlapped_csv['haloupdate0']
standard_csv['computation_time'] = standard_csv['elapsed_time'] - standard_csv['haloupdate0']
overlapped_heatmap_csv['computation_time'] = overlapped_heatmap_csv['elapsed_time'] - overlapped_heatmap_csv['haloupdate0']

def plot_elapsed_times_bars():
    for so in space_orders:
        overlapped_csv_so = overlapped_csv.loc[overlapped_csv['space_order'] == so]
        standard_csv_so = standard_csv.loc[standard_csv['space_order'] == so]

        def get_result_df(column_name):
            results = []
            for experiment_name in experiment_names:
                result = [experiment_name]
                for tts in time_tile_sizes:
                    result.append(overlapped_csv_so.loc[(overlapped_csv_so['time_tile_size'] == tts) & (overlapped_csv_so['experiment_name'] == experiment_name)][column_name].min())
                result.append(standard_csv_so.loc[standard_csv_so['experiment_name'] == experiment_name][column_name].min())
                results.append(result)
            results = pd.DataFrame(results)
            results.columns=['Dimensions', 'TTS=4', 'TTS=8', 'TTS=16', 'TTS=32', 'Standard MPI']
            return results 

        results = get_result_df('elapsed_time')
        results.plot(x='Dimensions', kind='bar', rot=10, ylabel="Time elapsed (s)",title="Elapsed Times, " +  platform_name + " Laplace Experiments, SO: " + str(so))
        plt.savefig(graphs_folder + "elapsed_time_" + str(so) + "so")

        results = get_result_df('computation_time')
        results.plot(x='Dimensions', kind='bar', rot=10, ylabel="Computation time (s)",title="Computation Times, " + platform_name + " Laplace Experiments, SO: " + str(so))
        plt.savefig(graphs_folder + "computation_time_" + str(so) + "so")

        results = get_result_df('haloupdate0')
        results.plot(x='Dimensions', kind='bar', rot=10, ylabel="Communication time (s)",title="Communication Times, " + platform_name + " Laplace Experiments, SO: " + str(so))
        plt.savefig(graphs_folder + "communication_time_" + str(so) + "so")

def plot_heatmaps():
    include_16_width = False
    experiment_name = "t=256,d=(256,256,256)"
    for so in space_orders:
        overlapped_csv_so = overlapped_heatmap_csv.loc[overlapped_heatmap_csv['space_order'] == so]
        for tts in time_tile_sizes:
            plt.clf()
            overlapped_csv_so_tts = overlapped_csv_so.loc[(overlapped_csv_so['time_tile_size'] == tts) & (include_16_width | ((overlapped_csv_so['wf_x_width'] > 16) & (overlapped_csv_so['wf_y_width'] > 16)))]
            overlapped_csv_so_tts = overlapped_csv_so_tts.loc[(overlapped_csv_so_tts['wf_x_width'] != 16) | (overlapped_csv_so_tts['wf_y_width'] != 16)]
            ax = sns.heatmap(overlapped_csv_so_tts.pivot("wf_y_width","wf_x_width","elapsed_time"),cmap='RdYlGn_r', cbar_kws={'label': 'Time Elapsed (s)'})
            ax.set_title(platform_name + " Laplace, " + experiment_name + ", TTS: " + str(tts) + ", SO:" + str(so))
            ax.set_xlabel("Wavefront X Width")
            ax.set_ylabel("Wavefront Y Width")
            ax.invert_yaxis()
            fig = ax.get_figure()
            fig.savefig(heatmaps_folder + "heatmap_" + str(tts) + "tts_" + str(so) + "so")


def plot_comm_time_lines():
    colours = ['blue', 'green', 'darkorange']
    handles = []

    for so in space_orders:
        colour = colours[space_orders.index(so)]  
        handles.append(mpatches.Patch(color=colour, label="Space Order " + str(so)))

    standard_comm_times = np.array(standard_csv['haloupdate0'])
    for tts in time_tile_sizes:
        fig, ax = plt.subplots()
        ax.set_title("Communication Times, " + platform_name + " Laplace Experiments, TTS: " + str(tts))
        ax.set_xlabel("Time spent on communcation, standard MPI (s)")
        ax.set_ylabel("Time spent on communcation, overlapped tiling MPI (s)")
        for so in space_orders:
            colour = colours[space_orders.index(so)]
            overlapped_comm_times = overlapped_csv.loc[(overlapped_csv['time_tile_size'] == tts) & (overlapped_csv['space_order'] == so)]
            overlapped_comm_times = overlapped_comm_times.groupby(['num_ranks', 'space_order', 'time_tile_size', 'time', 'x_size', 'y_size', 'z_size']).min().reset_index()
            overlapped_comm_times = np.array(overlapped_comm_times['haloupdate0'])
            plt.scatter(x=standard_csv.loc[standard_csv['space_order'] == so]['haloupdate0'], y=overlapped_comm_times, color= colour)

        overlapped_comm_times = overlapped_csv.loc[(overlapped_csv['time_tile_size'] == tts)]
        overlapped_comm_times = overlapped_comm_times.groupby(['num_ranks', 'space_order', 'time_tile_size', 'time', 'x_size', 'y_size', 'z_size']).min().reset_index()
        overlapped_comm_times = np.array(overlapped_comm_times['haloupdate0'])

        m, c = np.polyfit(standard_comm_times, overlapped_comm_times, deg=1)
        plt.plot(standard_comm_times, standard_comm_times * m + c, color='red')
        
        #standard_comm_times_no_so = np.array(standard_csv.loc[standard_csv['space_order'] != 8]['haloupdate0'])
        #overlapped_comm_times = overlapped_csv.loc[(overlapped_csv['time_tile_size'] == tts) & (overlapped_csv['space_order'] != 8)]
        #overlapped_comm_times = overlapped_comm_times.groupby(['num_ranks', 'space_order', 'time_tile_size', 'time', 'x_size', 'y_size', 'z_size']).min().reset_index()
        #overlapped_comm_times = np.array(overlapped_comm_times['haloupdate0'])
        #m, c = np.polyfit(standard_comm_times_no_so, overlapped_comm_times, deg=1)
        #plt.plot(standard_comm_times_no_so, standard_comm_times_no_so * m + c, color='blue')

        plt.legend(handles=handles)
        plt.text(standard_comm_times[2], overlapped_comm_times[-1], 'y = ' + str(round(m, ndigits=2)) + "x + " + str(round(c, ndigits=2)), size=10, weight="bold")
        plt.savefig(graphs_folder + "comm_times_line_" + str(tts) + "tts")

plot_elapsed_times_bars()
plot_heatmaps()
plot_comm_time_lines()
