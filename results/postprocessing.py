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

def CPU_name(architecture):
    if architecture == "yam":
        return "Xeon v3"
    if architecture == "hpc":
        return "Xeon v2"
    raise Exception("Typo in: " + str(architecture))

def stencil_name(stencil):
    if stencil == "laplace":
        return "Laplace"
    if stencil == "wave":
        return "Wave"
    raise Exception("Typo in: " + str(stencil))       

def get_architecture_mpi_results(architecture, stencil):
    architecture_folder = results_folder + architecture
    #overlapped_csv = pd.read_csv(architecture_folder + "/results_overlapped_laplace_mpi.csv")
    standard_csv = pd.read_csv(architecture_folder + "/" + str(stencil) + "_standard_mpi.csv")

    # Average across repeats
    #overlapped_csv = overlapped_csv.groupby(['num_ranks', 'space_order', 'time_tile_size','wf_x_width','wf_y_width', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])
    standard_csv = standard_csv.groupby(['num_ranks', 'space_order', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])

    #overlapped_csv['experiment_name'] = overlapped_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
    standard_csv['experiment_name'] = standard_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)

    #overlapped_csv['computation_time'] = overlapped_csv['elapsed_time'] - overlapped_csv['haloupdate0']
    standard_csv['computation_time'] = standard_csv['elapsed_time'] - standard_csv['haloupdate0']
    #return overlapped_csv, standard_csv
    return standard_csv

def get_architecture_openmp_results(architecture, stencil):
    architecture_folder = results_folder + architecture
    wavefront_csv = pd.read_csv(architecture_folder + "/" + str(stencil) + "_wavefront_openmp.csv")
    standard_csv = pd.read_csv(architecture_folder + "/" + str(stencil) + "_standard_openmp.csv")

    # Average across repeats
    wavefront_csv = wavefront_csv.groupby(['space_order', 'time_tile_size','wf_x_width','wf_y_width', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])
    standard_csv = standard_csv.groupby(['space_order', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])

    wavefront_csv['experiment_name'] = wavefront_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
    standard_csv['experiment_name'] = standard_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)

    return wavefront_csv, standard_csv     

def plot_time_bars_openmp(architecture, stencil):
    wavefront_csv, standard_csv = get_architecture_openmp_results(architecture, stencil)
    architecture_folder = graphs_folder + architecture
    
    def get_result_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            wavefront_csv_so = wavefront_csv.loc[wavefront_csv['space_order'] == so]
            standard_csv_so = standard_csv.loc[standard_csv['space_order'] == so]
            result.append(wavefront_csv_so[column_name].min())
            result.append(standard_csv_so[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'Wavefront Tiling', 'Standard Devito Code']
        return results

    results = get_result_df("elapsed_time")
    print(results)
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time elapsed (s)",title="Elapsed Times, " + CPU_name(architecture) + " " + stencil_name(stencil) + " Experiments")
    plt.savefig(architecture_folder + "/" + stencil + "_elapsed_times_wavefront")

def plot_time_bars_MPI_vs_nonMPI(architecture, stencil):
    _, standard_csv = get_architecture_openmp_results(architecture, stencil)
    standard_mpi_csv = get_architecture_mpi_results(architecture, stencil)
    architecture_folder = graphs_folder + architecture
    
    def get_result_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            standard_mpi_csv_so = standard_mpi_csv.loc[(standard_mpi_csv['space_order'] == so)]
            standard_csv_so = standard_csv.loc[(standard_csv['space_order'] == so)]
            result.append(standard_mpi_csv_so[column_name].min())
            result.append(standard_csv_so[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'Devito with MPI', 'Devito without MPI']
        return results

    results = get_result_df("elapsed_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time elapsed (s)",title="Elapsed Times, " + CPU_name(architecture) + " " + stencil_name(stencil) + " Experiments")
    plt.savefig(architecture_folder + "/" + stencil + "_mpi_vs_no_mpi")

plot_time_bars_MPI_vs_nonMPI("yam", "laplace")
plot_time_bars_openmp("yam", "laplace")