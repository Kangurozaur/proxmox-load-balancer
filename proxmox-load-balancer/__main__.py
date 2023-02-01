import time

from matplotlib import pyplot as plt
from model import *
from data_interface import load_cluster_from_file, perform_migration, save_cluster_to_file, load_cluster_info
from load_balancer import balance_cluster
from util import plot_cluster_cpu_usage
import sys

def main():
    # print('in main')
    # args = sys.argv[1:]
    # print('count of args :: {}'.format(len(args)))
    # for arg in args:
    #     print('passed argument :: {}'.format(arg))
    start_time = time.time()
    #test_simulation()
    cluster = load_cluster_info("week", "average")
    #save_cluster_to_file(cluster, "proxmox-load-balancer/snapshots/data_week_2")
    
    cluster = load_cluster_from_file("proxmox-load-balancer/snapshots/data_week_2")

    #perform_migration(114, "pve-g2n6", "pve-g2n8")
    #test_simulation()
    # plot_cluster_cpu_usage(cluster)

    # old_score = cluster.get_cluster_score(0.10)
    # print(old_score)
    print(cluster.get_cluster_score())
    plot_cluster_cpu_usage(cluster, [])
    cluster = balance_cluster(cluster)

    print(cluster.get_cluster_score())
    plot_cluster_cpu_usage(cluster, [])

    print("Cluster data loaded in --- %s seconds ---" % round(time.time() - start_time,3))

    #plot_cluster_cpu_usage(cluster)
    # Plot all figures
    plt.show()

if __name__ == '__main__':
    main()