import csv 
import matplotlib.pyplot as plt
import pandas as pd

space_orders = [2, 4, 8]
tts = 2
num_repeats = 3

results_folder = "results/"
graphs_folder = "graphs/"

def generate_graph(space_order):
    file_path_standard = results_folder + "results_standard_mpi_" + str(space_order) + "so.csv"
    file_path_overlapped = results_folder + "results_overlapped_mpi_" + str(space_order) + "so.csv"
    file_path_openmp = results_folder + "results_openmp_" + str(space_order) + "so.csv" 

    standard_csv = csv.reader(open(file_path_standard))
    overlapped_csv = csv.reader(open(file_path_overlapped))
    openmp_csv = csv.reader(open(file_path_openmp))

    headers = next(standard_csv)
    _ = next(overlapped_csv)
    _ = next(openmp_csv)

    results = []
    proportion_results = []
    halo_results = []
    compute_results = []

    for standard_row in standard_csv:
        overlapped_row = next(overlapped_csv)
        openmp_row = next(openmp_csv)
        timesteps = int(standard_row[headers.index("time")])
        x_size = int(standard_row[headers.index("x_size")])
        y_size = int(standard_row[headers.index("y_size")])
        z_size = int(standard_row[headers.index("z_size")])

        standard_times = []
        overlapped_times = []
        openmp_times = []
        
        standard_proportions = []
        overlapped_proportions = []

        standard_halo = []
        overlapped_halo = []
        standard_compute = []
        overlapped_compute = []

        for i in range(1, num_repeats):
            try:
                elapsed = float(standard_row[headers.index("elapsed_time")])
                halo = float(standard_row[headers.index("haloupdate0")])
                standard_times.append(elapsed)
                standard_proportions.append(halo / elapsed)
                standard_halo.append(halo)
                standard_compute.append(elapsed - halo)
            except IndexError:
                pass
            try:
                elapsed = float(overlapped_row[headers.index("elapsed_time")])
                halo = float(overlapped_row[headers.index("haloupdate0")])
                overlapped_times.append(elapsed)
                overlapped_proportions.append(halo / elapsed)
                overlapped_halo.append(halo)
                overlapped_compute.append(elapsed - halo)
            except IndexError:
                pass
            try:
                elapsed = float(openmp_row[headers.index("elapsed_time")])
                openmp_times.append(elapsed)
            except IndexError:
                pass
            standard_row = next(standard_csv)
            overlapped_row = next(overlapped_csv)
            openmp_row = next(openmp_csv)

        def get_avg_time(times):
            return sum(times) / len(times)
        
        experiment_name = "t="+str(timesteps)+",d=("+str(x_size)+","+str(y_size)+","+str(z_size)+ ")"
        results.append([experiment_name, get_avg_time(overlapped_times), get_avg_time(standard_times), get_avg_time(openmp_times)])
        proportion_results.append([experiment_name, get_avg_time(overlapped_proportions), get_avg_time(standard_proportions)])
        halo_results.append([experiment_name, get_avg_time(overlapped_halo), get_avg_time(standard_halo)])
        compute_results.append([experiment_name, get_avg_time(overlapped_compute), get_avg_time(standard_compute)])

    def save_graph(results):
        title = "Small Laplace Experiments, TTS: " + str(tts) + ", SO: " + str(space_order)
        df = pd.DataFrame(results, columns=['Dimensions', 'Overlapped Tiling MPI', 'Standard MPI', 'OpenMP'])
        df.plot(x='Dimensions', kind='bar', rot=10, ylabel="Time elapsed (s)", 
                title="Elapsed Times, " + title)
        plt.savefig(graphs_folder + "results_" + str(space_order) + "so")

        df = pd.DataFrame(proportion_results, columns=['Dimensions', 'Overlapped Tiling MPI', 'Standard MPI'])
        df.plot(x='Dimensions', kind='bar', rot=10, ylabel="Proportion time spent on communication vs. flop (%)", title="Communication Time Proportions, " + title)
        plt.savefig(graphs_folder + "proportions_" + str(space_order) + "so")

        df = pd.DataFrame(halo_results, columns=['Dimensions', 'Overlapped Tiling MPI', 'Standard MPI'])
        df.plot(x='Dimensions', kind='bar', rot=10, ylabel="Time spent on communication (s)", title="Communication Times, " + title)
        plt.savefig(graphs_folder + "comm_times_" + str(space_order) + "so")

        df = pd.DataFrame(compute_results, columns=['Dimensions', 'Overlapped Tiling MPI', 'Standard MPI'])
        df.plot(x='Dimensions', kind='bar', rot=10, ylabel="Time spent on floating point operations (s) ", title="Floating Point Operation Times, " + title)
        plt.savefig(graphs_folder + "flop_times_" + str(space_order) + "so")
    
    save_graph(results)

for space_order in space_orders:
    generate_graph(space_order)
