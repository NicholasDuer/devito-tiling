import csv 
import matplotlib.pyplot as plt
import pandas as pd
from requests import get

space_orders = [2, 4, 8]
num_repeats = 3

results_folder = "results/"
graphs_folder = "graphs/"

def generate_graph(space_order):
    file_path_standard = results_folder + "results_standard_" + str(space_order) + "so.csv"
    file_path_overlapped = results_folder + "results_overlapped_" + str(space_order) + "so.csv"

    standard_csv = csv.reader(open(file_path_standard))
    overlapped_csv = csv.reader(open(file_path_overlapped))

    headers = next(standard_csv)
    _ = next(overlapped_csv)

    two_rank_results = []
    four_rank_results = []

    for standard_row in standard_csv:
        overlapped_row = next(overlapped_csv)
        num_ranks = int(standard_row[headers.index("num_ranks")])
        timesteps = int(standard_row[headers.index("time")])
        x_size = int(standard_row[headers.index("x_size")])
        y_size = int(standard_row[headers.index("y_size")])
        z_size = int(standard_row[headers.index("z_size")])

        standard_times = []
        overlapped_times = []

        for i in range(num_repeats):
            standard_times.append(float(standard_row[headers.index("elapsed_time")]))
            overlapped_times.append(float(overlapped_row[headers.index("elapsed_time")]))
            standard_row = next(standard_csv)
            overlapped_row = next(overlapped_csv)

        def get_avg_time(times):
            return sum(times) / len(times)
        
        experiment_name = "t="+str(timesteps)+",d=("+str(x_size)+","+str(y_size)+","+str(z_size)+ ")"
        if (num_ranks == 2):
            two_rank_results.append([experiment_name, get_avg_time(standard_times), get_avg_time(overlapped_times)])
            
        if (num_ranks == 4):
            four_rank_results.append([experiment_name, get_avg_time(standard_times), get_avg_time(overlapped_times)])
    
    def save_graph(results, num_ranks):
        df = pd.DataFrame(results, columns=['Dimensions', 'Standard MPI', 'Overlapped Tiling MPI'])
        df.plot(x='Dimensions', kind='bar', rot=10, ylabel="Average Time elapsed (s)", 
                title="Laplace Experiments, Space Order: " + str(space_order) + ", Ranks: " + str(num_ranks))
        plt.savefig(graphs_folder + "results_ " + str(space_order) + "so_" + str(num_ranks) + "_ranks")
    
    save_graph(two_rank_results, num_ranks=2)
    save_graph(four_rank_results, num_ranks=4)

for space_order in space_orders:
    generate_graph(space_order)
