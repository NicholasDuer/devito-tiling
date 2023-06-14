from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib
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
    overlapped_csv = pd.read_csv(architecture_folder + "/" + str(stencil) + "_overlapped_mpi.csv")
    standard_csv = pd.read_csv(architecture_folder + "/" + str(stencil) + "_standard_mpi.csv")

    # Average across repeats
    overlapped_csv = overlapped_csv.groupby(['num_ranks', 'space_order', 'time_tile_size','wf_x_width','wf_y_width', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])
    standard_csv = standard_csv.groupby(['num_ranks', 'space_order', 'time', 'x_size', 'y_size', 'z_size']).mean().reset_index().drop(columns=['repeat_num'])

    overlapped_csv['experiment_name'] = overlapped_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)
    standard_csv['experiment_name'] = standard_csv.apply(lambda x: "t=" + str(int(x['time'])) + ",d=(" + str(int(x['x_size'])) + "," + str(int(x['y_size'])) + "," + str(int(x['z_size'])) + ")", axis=1)

    overlapped_csv['computation_time'] = overlapped_csv['elapsed_time'] - overlapped_csv['haloupdate0']
    standard_csv['computation_time'] = standard_csv['elapsed_time'] - standard_csv['haloupdate0']
    return overlapped_csv, standard_csv

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

def plot_wavefront_bars(architecture, stencil):
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
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time Elapsed (s)",title="Elapsed Times, " + CPU_name(architecture) + " " + stencil_name(stencil) + " Experiments", zorder=3)
    plt.grid(axis='y', zorder=0)
    plt.savefig(architecture_folder + "/" + stencil + "/" + "elapsed_times_wavefront")

    def get_result_tts_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            overlapped_csv_so = wavefront_csv.loc[(wavefront_csv['space_order'] == so)]
            for tts in time_tile_sizes:
                overlapped_csv_tts = overlapped_csv_so.loc[overlapped_csv_so['time_tile_size'] == tts]
                if (column_name == "gpointss"):
                    result.append(overlapped_csv_tts[column_name].max())
                elif (column_name == "haloupdate0"):
                    result.append(overlapped_csv_tts[column_name].mean())
                else:
                    result.append(overlapped_csv_tts[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'TTH=4', 'TTH=8', 'TTH=16', 'TTH=32']
        return results

    results = get_result_tts_df("elapsed_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time Elapsed (s)",title="Elapsed Times, " +  CPU_name(architecture) + " Laplace Experiments", zorder=3)
    plt.grid(axis='y', zorder=0)
    plt.savefig(architecture_folder + "/" + stencil + "/elapsed_times_wavefront_per_tts")       

def plot_time_bars_MPI_vs_nonMPI(architecture, stencil):
    _, standard_csv = get_architecture_openmp_results(architecture, stencil)
    _ ,standard_mpi_csv = get_architecture_mpi_results(architecture, stencil)
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
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time Elapsed (s)",title="Elapsed Times, " + CPU_name(architecture) + " " + stencil_name(stencil) + " Experiments", zorder=3)
    plt.grid(axis='y', zorder=0)
    plt.savefig(architecture_folder + "/" + stencil + "/" + "mpi_vs_no_mpi")

def plot_overlapped_bars(architecture, stencil):
    overlapped_csv, standard_csv = get_architecture_mpi_results(architecture, stencil)
    architecture_folder = graphs_folder + architecture
    
    def get_result_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            overlapped_csv_so = overlapped_csv.loc[(overlapped_csv['space_order'] == so)]
            standard_csv_so = standard_csv.loc[(standard_csv['space_order'] == so)]
            if (column_name == "gpointss"):
                result.append(overlapped_csv_so[column_name].max())
                result.append(standard_csv_so[column_name].max())
            elif (column_name == "haloupdate0"):
                result.append(overlapped_csv_so[column_name].mean())  
                result.append(standard_csv_so[column_name].mean())  
            else:
                result.append(overlapped_csv_so[column_name].min())
                result.append(standard_csv_so[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'Overlapped Tiling MPI', 'Standard MPI']
        return results

    results = get_result_df("elapsed_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time Elapsed (s)",title="Elapsed Times, " + CPU_name(architecture) + " Laplace Experiments", zorder=3)
    plt.grid(axis='y', zorder=0)
    plt.savefig(architecture_folder + "/" + stencil + "/elapsed_times_mpi")

    results = get_result_df("gpointss")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="GPoints/s",title="GPoints/s, " +  CPU_name(architecture) + " Laplace Experiments", zorder=3)
    plt.grid(axis='y', zorder=0)
    plt.savefig(architecture_folder + "/" + stencil + "/gpointss_mpi")

    results = get_result_df("computation_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Computation Time (s)",title="Computation Times, " +  CPU_name(architecture) + " Laplace Experiments", zorder=3)
    plt.grid(axis='y', zorder=0)
    plt.savefig(architecture_folder + "/" + stencil + "/computation_times")

    def get_result_df_across_tts(column_name):
        results = []
        for tts in time_tile_sizes:
            result = ["tts=" + str(tts)]
            overlapped_csv_tts = overlapped_csv.loc[(overlapped_csv['time_tile_size'] == tts)]
            result.append(overlapped_csv_tts[column_name].mean())  
            result.append(standard_csv[column_name].mean())  
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Time Tile Size', 'Overlapped Tiling MPI', 'Standard MPI']
        return results

    results = get_result_df("haloupdate0")
    print(results)
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="MPI Communication Times (s)",title="Communication Times, " +  CPU_name(architecture) + " Laplace Experiments", zorder=3)
    plt.grid(axis='y', zorder=0)
    plt.savefig(architecture_folder + "/" + stencil + "/communication_times")

    def get_result_tts_df(column_name):
        results = []
        for so in space_orders:
            result = ["so=" + str(so)]
            overlapped_csv_so = overlapped_csv.loc[(overlapped_csv['space_order'] == so)]
            standard_csv_so = standard_csv.loc[(standard_csv['space_order']) == so]
            for tts in time_tile_sizes:
                overlapped_csv_tts = overlapped_csv_so.loc[overlapped_csv_so['time_tile_size'] == tts]
                if (column_name == "gpointss"):
                    result.append(overlapped_csv_tts[column_name].max())
                elif (column_name == "haloupdate0"):
                    result.append(overlapped_csv_tts[column_name].mean())
                else:
                    result.append(overlapped_csv_tts[column_name].min())
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ['Space Order', 'TTH=4', 'TTH=8', 'TTH=16', 'TTH=32']
        return results
    
    results = get_result_tts_df("elapsed_time")
    results.plot(x='Space Order', kind='bar', rot=0, ylabel="Time Elapsed (s)",title="Elapsed Times, " +  CPU_name(architecture) + " Laplace Experiments", zorder=3)
    plt.grid(axis='y', zorder=0)
    plt.savefig(architecture_folder + "/" + stencil + "/elapsed_times_per_tts_mpi")

def plot_heatmaps_MPI(architecture, stencil):
    font = {'family' : 'normal',
        'size'   : 24}

    matplotlib.rc('font', **font)
    overlapped_csv, _ = get_architecture_mpi_results(architecture, stencil)
    heatmaps_folder = graphs_folder + architecture + "/" + stencil + "/heatmaps"
    
    include_16_width = False

    for so in space_orders:
        overlapped_csv_so = overlapped_csv.loc[overlapped_csv['space_order'] == so]
        for tts in time_tile_sizes:
            plt.clf()
            fig = plt.figure(figsize=(8, 8))
            overlapped_csv_so_tts = overlapped_csv_so.loc[(overlapped_csv_so['time_tile_size'] == tts) & (include_16_width | ((overlapped_csv_so['wf_x_width'] > 16) & (overlapped_csv_so['wf_y_width'] > 16)))]
            ax = sns.heatmap(overlapped_csv_so_tts.pivot("wf_y_width","wf_x_width","elapsed_time"),cmap='RdYlGn_r', cbar=False)
            print("tts: " + str(tts) + " so " + str(so))
            print(overlapped_csv_so_tts.pivot("wf_y_width","wf_x_width","elapsed_time"))
            ax.set_title("so=" + str(so))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.invert_yaxis()
            fig.savefig(heatmaps_folder + "/heatmap_" + str(tts) + "tth_" + str(so) + "so")

font = {'size': 12}

matplotlib.rc('font', **font)
#plot_time_bars_MPI_vs_nonMPI("yam", "laplace")
plot_wavefront_bars("yam", "laplace")
plot_overlapped_bars("hpc", "laplace")
#plot_heatmaps_MPI("hpc", "laplace")