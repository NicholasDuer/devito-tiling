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
    overlapped_csv = pd.read_csv(architecture_folder + "/results_overlapped_mpi.csv")
    standard_csv = pd.read_csv(architecture_folder + "/results_standard_mpi.csv")

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
    return overlapped_csv, standard_csv

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

def plot_elapsed_time_bars_MPI(architecture):
    overlapped_csv, standard_csv = get_architecture_mpi_results(architecture)
    experiment_name = "t=256,d=(512,512,512)"
    architecture_folder = graphs_folder + architecture
    
    def get_result_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            overlapped_csv_so = overlapped_csv.loc[(overlapped_csv['experiment_name'] == experiment_name) & (overlapped_csv['space_order'] == so)]
            standard_csv_so = standard_csv.loc[(standard_csv['experiment_name'] == experiment_name) & (standard_csv['space_order'] == so)]
            result.append(overlapped_csv_so[column_name].min())
            result.append(standard_csv_so[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'Wavefront MPI+OpenMP', 'Standard MPI+OpenMP']
        return results

    results = get_result_df("elapsed_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time elapsed (s)",title="Elapsed Times, " +  architecture.upper() + " Laplace Experiments")
    print(results)

    # Add percentage labels
    for index, row in results.iterrows():
        wf = row['Wavefront MPI+OpenMP']
        std = row['Standard MPI+OpenMP']
        plt.text(wf / std, wf, std)

    plt.savefig(architecture_folder + "/elasped_times_mpi")

def plot_elapsed_time_bars_openmp(architecture):
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
        results.columns = ['Space Order', 'Wavefront OpenMP', 'Standard OpenMP']
        return results

    results = get_result_df("elapsed_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time elapsed (s)",title="Elapsed Times, " +  architecture.upper() + " Laplace Experiments")
    plt.savefig(architecture_folder + "/elasped_times_openmp")

plot_elapsed_time_bars_MPI("yam")
plot_elapsed_time_bars_MPI("hpc")
plot_elapsed_time_bars_openmp("hpc")

