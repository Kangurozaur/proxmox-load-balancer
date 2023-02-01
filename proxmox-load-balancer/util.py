
from copy import deepcopy
from math import fsum
from matplotlib import pyplot as plt
import numpy as np
from model import Cluster
import matplotlib.dates as mdate
import matplotlib.ticker as mtick


def plot_cluster_cpu_usage(cluster: Cluster, node_names_list = []):
    
    # Sort cluster nodes by name

    cluster.nodes.sort(key=lambda a: a.name)

    n_nodes = len(cluster.nodes)

    # If node_names specified, check if node_names specified exist in cluster
    if node_names_list != []:
        node_names = cluster.get_node_names()
        for node_name in node_names_list:
            if node_name not in node_names:
                raise Exception("Node with name: {0} does not exist in cluster: {1}".format(node_name, cluster.name))
        n_nodes = len(node_names_list)
    
    
    fig, ax = plt.subplots(n_nodes+1,1)

    xpoints = []
    y_sum = []
    y_sum_ceiling = []
    # Subplot index
    plot_index = 0
    for i in range(0, len(cluster.nodes)):
        if node_names_list == [] or cluster.nodes[i].name in node_names_list:
            # Calculate aggregated CPU utilization
            if plot_index == 0:
                (xpoints,y_sum) = cluster.nodes[i].plot(fig, ax[plot_index], 1)
                
                y_temp = deepcopy(y_sum)
                y_sum_ceiling = y_temp
            else:
                (_ ,y_temp) = cluster.nodes[i].plot(fig, ax[plot_index], 1)
                y_sum = list(map(lambda a,b: a + b, y_sum, y_temp))
                y_sum_ceiling = list(map(lambda a,b: a + b, y_sum_ceiling, y_temp))
            plot_index = plot_index + 1
    y_average = list(map(lambda a: a/n_nodes, y_sum))
    y_average_ceiling = list(map(lambda a: a/n_nodes, y_sum_ceiling))
    # Plot average
    ax[n_nodes].set_title("Average cluster load", fontstyle='italic')
    ax[n_nodes].plot_date(xpoints, y_average_ceiling, '-o')

    # Format to show dates
    date_fmt = '%d-%m-%y %H:%M:%S'
    date_formatter = mdate.DateFormatter(date_fmt)
    ax[n_nodes].xaxis.set_major_formatter(date_formatter)
    

    # Format to show %
    ax[n_nodes].yaxis.set_major_formatter(mtick.PercentFormatter(1.0,1))

    fig.autofmt_xdate()
    total_average = fsum(y_average_ceiling)/len(y_average_ceiling)
    print("Average cluster utilization is: {0}".format(total_average))
    ax[n_nodes].axhline(y = total_average, color = 'y', linestyle = '-')
    ax[n_nodes].axhline(y = 1, color = 'r', linestyle = '-')

def compare_cluster_states(cluster1, cluster2):
    reference_cluster = deepcopy(cluster1)
    simulated_cluster = deepcopy(cluster2)

def trim_cluster_data(cluster, timeframe):
    start, end = timeframe
    for node in cluster.nodes:
        node.utilization = node.utilization[start:end]
        node.aggregate_utilization = node.aggregate_utilization[start:end]
        for vm in node.vms:
            vm.utilization = vm.utilization[start:end]
            
    return cluster

def pearson_coefficient(data1, data2):

    if len(data1) != len(data2):
        raise Exception("Datasets must be of the same length to find corresponding points.")

    # Calculate means of datasets
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)

    # Calculate standard deviations of datasets
    std1 = np.std(data1)
    std2 = np.std(data2)

    # Calculate Pearson correlation coefficient
    r = np.sum([(d1 - mean1) * (d2 - mean2) for d1, d2 in zip(data1, data2)]) / (len(data1) * std1 * std2)

    return r