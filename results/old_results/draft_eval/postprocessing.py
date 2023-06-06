from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns

results_folder = "data/"
graphs_folder = "graphs/"

space_orders = [2,4,8]
time_tile_sizes = [4,8,16,32]

def get_architecture_mpi_results(architecture):
    architecture_folder = results_folder + architecture
    overlapped_csv = pd.read_csv(architecture_folder + "/results_overlapped_laplace_mpi.csv")
    standard_csv = pd.read_csv(architecture_folder + "/results_standard_laplace_mpi.csv")

    # Average across repeats
    overlapped_csv = overlapped_csv.groupby(['num_ranks', 'space_order', 'time_tile_size','wf_x_width','wf_y_width', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])
    standard_csv = standard_csv.groupby(['num_ranks', 'space_order', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])

    overlapped_csv['experiment_name'] = overlapped_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
    standard_csv['experiment_name'] = standard_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
    experiment_names = overlapped_csv['experiment_name'].unique()

    overlapped_csv['computation_time'] = overlapped_csv['elapsed_time'] - overlapped_csv['haloupdate0']
    standard_csv['computation_time'] = standard_csv['elapsed_time'] - standard_csv['haloupdate0']
    return overlapped_csv, standard_csv

def get_architecture_mpi_heatmap_results(architecture):
    architecture_folder = results_folder + architecture
    overlapped_csv = pd.read_csv(architecture_folder + "/results_overlapped_heatmaps_mpi.csv")

    # Average across repeats
    overlapped_csv = overlapped_csv.groupby(['num_ranks', 'space_order', 'time_tile_size','wf_x_width','wf_y_width', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])

    overlapped_csv['experiment_name'] = overlapped_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)

    overlapped_csv['computation_time'] = overlapped_csv['elapsed_time'] - overlapped_csv['haloupdate0']
    return overlapped_csv

def get_architecture_openmp_results(architecture):
    architecture_folder = results_folder + architecture + "_no_mpi"
    wavefront_csv = pd.read_csv(architecture_folder + "/results_wavefront_openmp.csv")
    standard_csv = pd.read_csv(architecture_folder + "/results_standard_openmp.csv")

    # Average across repeats
    wavefront_csv = wavefront_csv.groupby(['space_order', 'time_tile_size','wf_x_width','wf_y_width', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])
    standard_csv = standard_csv.groupby(['space_order', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])

    wavefront_csv['experiment_name'] = wavefront_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
    standard_csv['experiment_name'] = standard_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
    experiment_names = wavefront_csv['experiment_name'].unique()
    last_experiment = experiment_names[-1]
    experiment_names[-1] = experiment_names[-2]
    experiment_names[-2] = last_experiment

    return wavefront_csv, standard_csv  

def plot_time_bars_MPI(architecture):
    overlapped_csv, standard_csv = get_architecture_mpi_results(architecture)
    experiment_name = "t=256,d=(512,512,512)"
    architecture_folder = graphs_folder + architecture
    
    def get_result_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            overlapped_csv_so = overlapped_csv.loc[(overlapped_csv['experiment_name'] == experiment_name) & (overlapped_csv['space_order'] == so)]
            standard_csv_so = standard_csv.loc[(standard_csv['experiment_name'] == experiment_name) & (standard_csv['space_order'] == so)]
            if (column_name == "gpointss"):
                result.append(overlapped_csv_so[column_name].max())
                result.append(standard_csv_so[column_name].max())
            else:
                result.append(overlapped_csv_so[column_name].min())
                result.append(standard_csv_so[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'Wavefront MPI+OpenMP', 'Standard MPI+OpenMP']
        return results

    results = get_result_df("elapsed_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time elapsed (s)",title="Elapsed Times, " +  architecture.upper() + " Laplace Experiments")
    plt.savefig(architecture_folder + "/elapsed_times_mpi")

    results = get_result_df("gpointss")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="GPoints/s",title="GPoints/s, " +  architecture.upper() + " Laplace Experiments")
    plt.savefig(architecture_folder + "/gpointss_mpi")

    results = get_result_df("haloupdate0")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="MPI Communication Times",title="Communication Times, " +  architecture.upper() + " Laplace Experiments")
    plt.savefig(architecture_folder + "/communication_times")

    def get_result_tts_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            overlapped_csv_so = overlapped_csv.loc[(overlapped_csv['experiment_name'] == experiment_name) & (overlapped_csv['space_order'] == so)]
            for tts in time_tile_sizes:
                overlapped_csv_tts = overlapped_csv_so.loc[overlapped_csv_so['time_tile_size'] == tts]
                if (column_name == "gpointss"):
                    result.append(overlapped_csv_tts[column_name].max())
                else:
                    result.append(overlapped_csv_tts[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'TTS=4', 'TTS=8', 'TTS=16', 'TTS=32']
        return results
    
    results = get_result_tts_df("elapsed_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time elapsed (s)",title="Elapsed Times, " +  architecture.upper() + " Laplace Experiments")
    plt.savefig(architecture_folder + "/elapsed_times_per_tts_mpi")   

def plot_time_bars_openmp(architecture):
    wavefront_csv, standard_csv = get_architecture_openmp_results(architecture)
    experiment_name = "t=256,d=(512,512,512)"
    architecture_folder = graphs_folder + architecture
    
    def get_result_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            wavefront_csv_so = wavefront_csv.loc[(wavefront_csv['experiment_name'] == experiment_name) & (wavefront_csv['space_order'] == so)]
            standard_csv_so = standard_csv.loc[(standard_csv['experiment_name'] == experiment_name) & (standard_csv['space_order'] == so)]
            result.append(wavefront_csv_so[column_name].min())
            result.append(standard_csv_so[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'Wavefront Tiling', 'Standard Devito Code']
        return results

    results = get_result_df("elapsed_time")
    print(results)
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time elapsed (s)",title="Elapsed Times, Xeon v2 Laplace Experiments")
    plt.savefig(architecture_folder + "/elapsed_times_openmp")

    def get_result_tts_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            wavefront_csv_so = wavefront_csv.loc[(wavefront_csv['experiment_name'] == experiment_name) & (wavefront_csv['space_order'] == so)]
            for tts in time_tile_sizes:
                overlapped_csv_tts = wavefront_csv_so.loc[wavefront_csv_so['time_tile_size'] == tts]
                if (column_name == "gpointss"):
                    result.append(overlapped_csv_tts[column_name].max())
                else:
                    result.append(overlapped_csv_tts[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'TTS=4', 'TTS=8', 'TTS=16', 'TTS=32']
        return results
    
    results = get_result_tts_df("elapsed_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time elapsed (s)",title="Elapsed Times, " +  architecture.upper() + " Laplace Experiments")
    plt.savefig(architecture_folder + "/elapsed_times_per_tts_openmp")   

def plot_time_bars_MPI_vs_nonMPI(architecture):
    _, standard_csv = get_architecture_openmp_results(architecture)
    _, standard_mpi_csv = get_architecture_mpi_results(architecture)
    experiment_name = "t=256,d=(512,512,512)"
    architecture_folder = graphs_folder + architecture
    
    def get_result_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            standard_mpi_csv_so = standard_mpi_csv.loc[(standard_mpi_csv['experiment_name'] == experiment_name) & (standard_mpi_csv['space_order'] == so)]
            standard_csv_so = standard_csv.loc[(standard_csv['experiment_name'] == experiment_name) & (standard_csv['space_order'] == so)]
            result.append(standard_mpi_csv_so[column_name].min())
            result.append(standard_csv_so[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'Devito with MPI', 'Devito without MPI']
        return results

    results = get_result_df("elapsed_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time elapsed (s)",title="Elapsed Times, Xeon v2 Laplace Experiments")
    plt.savefig(architecture_folder + "/elapsed_times_mpi_vs_no_mpi")

def plot_heatmaps_MPI(architecture):
    overlapped_csv = get_architecture_mpi_heatmap_results(architecture)
    include_16_width = False
    experiment_name = "t=256,d=(256,256,256)"
    heatmaps_folder = graphs_folder + architecture + "/heatmaps"

    for so in space_orders:
        overlapped_csv_so = overlapped_csv.loc[overlapped_csv['space_order'] == so]
        for tts in time_tile_sizes:
            plt.clf()
            overlapped_csv_so_tts = overlapped_csv_so.loc[(overlapped_csv_so['time_tile_size'] == tts) & (include_16_width | ((overlapped_csv_so['wf_x_width'] > 16) & (overlapped_csv_so['wf_y_width'] > 16)))]
            ax = sns.heatmap(overlapped_csv_so_tts.pivot("wf_y_width","wf_x_width","elapsed_time"),cmap='RdYlGn_r', cbar_kws={'label': 'Time Elapsed (s)'})
            ax.set_title(architecture.upper() + " Laplace, " + experiment_name + ", TTS: " + str(tts) + ", SO:" + str(so))
            ax.set_xlabel("Wavefront X Width")
            ax.set_ylabel("Wavefront Y Width")
            ax.invert_yaxis()
            fig = ax.get_figure()
            fig.savefig(heatmaps_folder + "/heatmap_" + str(tts) + "tts_" + str(so) + "so")

#plot_time_bars_MPI("yam")
#plot_time_bars_MPI("hpc")
plot_time_bars_openmp("hpc")
#plot_time_bars_MPI_vs_nonMPI("hpc")
#plot_heatmaps_MPI("hpc")
#plot_heatmaps_MPI("yam")