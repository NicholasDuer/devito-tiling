import csv 
import matplotlib.pyplot as plt
import pandas as pd
from requests import get

space_orders = [2, 4, 8]
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

        for i in range(num_repeats):
            try:
                standard_times.append(float(standard_row[headers.index("elapsed_time")]))
            except IndexError:
                pass
            try:
                overlapped_times.append(float(overlapped_row[headers.index("elapsed_time")]))
            except IndexError:
                pass
            try:
                openmp_times.append(float(openmp_row[headers.index("elapsed_time")]))
            except IndexError:
                pass
            standard_row = next(standard_csv)
            overlapped_row = next(overlapped_csv)
            openmp_row = next(openmp_csv)

        def get_avg_time(times):
            return sum(times) / len(times)
        
        experiment_name = "t="+str(timesteps)+",d=("+str(x_size)+","+str(y_size)+","+str(z_size)+ ")"
        results.append([experiment_name, get_avg_time(overlapped_times), get_avg_time(standard_times), get_avg_time(openmp_times)])
    
    def save_graph(results):
        df = pd.DataFrame(results, columns=['Dimensions', 'Overlapped Tiling MPI', 'Standard MPI', 'OpenMP'])
        df.plot(x='Dimensions', kind='bar', rot=10, ylabel="Average Time elapsed (s)", 
                title="Laplace Experiments, Time Tile Size: 8, Space Order: " + str(space_order))
        plt.savefig(graphs_folder + "results_" + str(space_order) + "so")
    
    save_graph(results)

for space_order in space_orders:
    generate_graph(space_order)
