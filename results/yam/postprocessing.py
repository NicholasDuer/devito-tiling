from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

results_folder = "results/"
graphs_folder = "graphs/"
heatmaps_folder = graphs_folder + "heatmaps/"

space_orders = [2,4,8]
time_tile_sizes = [4,8,16,32]

overlapped_csv = pd.read_csv(results_folder + "./results_overlapped_mpi.csv")
standard_csv = pd.read_csv(results_folder + "/results_standard_mpi.csv")

# Average across repeats
overlapped_csv = overlapped_csv.groupby(['num_ranks', 'space_order', 'time_tile_size','wf_x_width','wf_y_width', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])
standard_csv = standard_csv.groupby(['num_ranks', 'space_order', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])

overlapped_csv['experiment_name'] = overlapped_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
standard_csv['experiment_name'] = standard_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
experiment_names = overlapped_csv['experiment_name'].unique()
last_experiment = experiment_names[-1]
experiment_names[-1] = experiment_names[-2]
experiment_names[-2] = last_experiment

overlapped_csv['computation_time'] = overlapped_csv['elapsed_time'] - overlapped_csv['haloupdate0']
standard_csv['computation_time'] = standard_csv['elapsed_time'] - standard_csv['haloupdate0']

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
        results.plot(x='Dimensions', kind='bar', rot=10, ylabel="Time elapsed (s)",title="Elapsed Times, YAM Laplace Experiments, SO: " + str(so))
        plt.savefig(graphs_folder + "elapsed_time_" + str(so) + "so")

        results = get_result_df('computation_time')
        results.plot(x='Dimensions', kind='bar', rot=10, ylabel="Computation time (s)",title="Computation Times, YAM Laplace Experiments, SO: " + str(so))
        plt.savefig(graphs_folder + "computation_time_" + str(so) + "so")

def plot_heatmaps():
    for so in space_orders:
        overlapped_csv_so = overlapped_csv.loc[overlapped_csv['space_order'] == so]
        
        for experiment_name in experiment_names:
            overlapped_csv_so_exp = overlapped_csv_so[overlapped_csv_so['experiment_name'] == experiment_name]
            x = overlapped_csv_so_exp['wf_x_width'].tolist()
            y = overlapped_csv_so_exp['wf_y_width'].tolist()
            z = overlapped_csv_so_exp['time_tile_size'].tolist()
            c = overlapped_csv_so_exp['computation_time'].tolist()
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')

            # configuring colorbar
            color_map = ScalarMappable()
            color_map.set_array(c)
            
            # creating the heatmap
            img = ax.scatter(x, y, z, marker='s', s=200)
            plt.colorbar(color_map)

            ax.set_title("Wavefront Parameter Search Space, " + experiment_name + ", SO: " + str(so))
            ax.set_xlabel("Wavefront X Width")
            ax.set_ylabel("Wavefront Y Width")
            ax.set_zlabel("Wavefront Height")
            plt.savefig(heatmaps_folder + "heatmap_" + experiment_name + "_" + str(so) + "so")
            plt.close(fig)

def plot_comm_time_lines():
    colours = ['blue', 'green', 'darkorange']
    handles = []

    for so in space_orders:
        colour = colours[space_orders.index(so)]  
        handles.append(mpatches.Patch(color=colour, label="Space Order " + str(so)))


    standard_comm_times = np.array(standard_csv['haloupdate0'])
    for tts in time_tile_sizes:
        fig, ax = plt.subplots()
        ax.set_title("Communication Times, Laplace Experiments, TTS: " + str(tts))
        ax.set_xlabel("Time spent on communcation, standard MPI (s)")
        ax.set_ylabel("Time spent on communcation, overlapped tiling MPI (s)")
        for so in space_orders:
            colour = colours[space_orders.index(so)]  
            overlapped_comm_times = np.array(overlapped_csv.loc[(overlapped_csv['time_tile_size'] == tts) & (overlapped_csv['space_order'] == so)]['haloupdate0'])
            plt.scatter(x=standard_csv.loc[standard_csv['space_order'] == so]['haloupdate0'], y=overlapped_comm_times, color= colour)

        overlapped_comm_times = np.array(overlapped_csv.loc[(overlapped_csv['time_tile_size'] == tts)]['haloupdate0'])
        m, c = np.polyfit(standard_comm_times, overlapped_comm_times, deg=1)
        plt.plot(standard_comm_times, standard_comm_times * m + c, color='red')
        plt.legend(handles=handles)
        plt.text(standard_comm_times[10], overlapped_comm_times[-4], 'y = ' + str(round(m, ndigits=2)) + "x + " + str(round(c, ndigits=2)), size=10, weight="bold")
        plt.savefig(graphs_folder + "comm_times_line_" + str(tts) + "tts")

plot_elapsed_times_bars()
plot_heatmaps()
#plot_comm_time_lines()
