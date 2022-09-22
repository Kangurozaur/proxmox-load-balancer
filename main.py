from copy import deepcopy
from datetime import datetime
from math import floor
from proxmoxer import ProxmoxAPI
import yaml
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.ticker as mtick
import numpy as np


class VmUsageRecord:

    def __init__(self, time, disk, maxdisk, cpu, maxcpu, netin, netout, diskread, diskwrite, mem, maxmem):
        self.time = time
        self.disk = disk
        self.maxdisk = maxdisk
        self.cpu = cpu
        self.maxcpu = maxcpu
        self.netin = netin
        self.netout = netout
        self.diskread = diskread
        self.diskwrite = diskwrite
        self.mem = mem
        self.maxmem = maxmem

class NodeUsageRecord:

    def __init__(self, time, maxcpu, netout, netin, cpu, swapused, swaptotal, memtotal, memused, loadavg, iowait, rootused, roottotal):
        self.time = time
        self.maxcpu = maxcpu
        self.netout = netout
        self.netin = netin
        self.swapused = swapused
        self.swaptotal = swaptotal
        self.memtotal = memtotal
        self.memused = memused
        self.loadavg = loadavg
        self.iowait = iowait
        self.rootused =  rootused
        self.cpu = cpu
        self.roottotal = roottotal

class VM:

    def __init__(self, id):
        self.id = id
        self.utilization = []

    def add_usage_record(record:VmUsageRecord):
        self.utilization.add(record)

class Node:

    def __init__(self, name = ""):
        self.name = name
        self.vms = []
        self.utilization = []
    
    def add_usage_record(self, record:NodeUsageRecord):
        self.utilization.add(record)

    def plot(self, fig, ax, n_aggregate=1):
        # Plot Node usage records
        n_time_points = len(self.utilization)
        xpoints = []
        ypoints = []

        for i in range(0, floor(n_time_points/n_aggregate)):
            time_sum = 0.0
            sum = 0.0

            for j in range(0, n_aggregate):
                #print("\t{0}".format(datetime.fromtimestamp( self.utilization[i*n_aggregate + j].time )))
                time_sum += self.utilization[i*n_aggregate + j].time
                
                for vm in self.vms:
                    sum += vm.utilization[i*n_aggregate + j].cpu
            
            #print(datetime.fromtimestamp( time_sum/n_aggregate ))
            xpoints.append(mdate.epoch2num(time_sum/n_aggregate))

            ypoints.append(sum/n_aggregate)

        ax.set_title(self.name, fontstyle='italic')
        ax.plot_date(xpoints, ypoints, '-o')

        # Format to show dates
        date_fmt = '%d-%m-%y %H:%M:%S'
        date_formatter = mdate.DateFormatter(date_fmt)
        ax.xaxis.set_major_formatter(date_formatter)
        

        # Format to show %
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0,1))

        fig.autofmt_xdate()
        ax.axhline(y = 1, color = 'r', linestyle = '-')

class Cluster:
    def __init__(self, name):
        self.name = name
        self.nodes = []

def load_cluster_info() -> Cluster:

    cluster = Cluster("cluster_1")

    # Define interesting metrics
    metrics = [
        "memory",
        "cores",
        "ostype",
        "bootdisk",
        "cpuunits",
        "cpulimit",
        "cpu"
    ]

    # Load config file
    config = yaml.safe_load(open("config.yaml"))

    # Initialize ProxmoxAPI

    proxmox = ProxmoxAPI(config["connection"]["url"]["ip"], user=config["connection"]["auth"]["username"], password=config["connection"]["auth"]["password"], verify_ssl=False)

    nodes_list = proxmox.nodes.get()
    for node in nodes_list:
        
        current_node = Node(node["node"])

        # Gather and add node utilization history
        node_usage_data = proxmox.nodes(node["node"]).rrddata.get(timeframe="week", cf="AVERAGE")
        for data_entry in node_usage_data:
            current_node.utilization.append(
                NodeUsageRecord(
                    data_entry["time"],
                    data_entry["maxcpu"],
                    data_entry["netout"],
                    data_entry["netin"],
                    data_entry["swapused"],
                    data_entry["swaptotal"],
                    data_entry["memtotal"],
                    data_entry["memused"],
                    data_entry["loadavg"],
                    data_entry["iowait"],
                    data_entry["rootused"],
                    data_entry["cpu"],
                    data_entry["roottotal"]
                )
            )

        # Gather assignment of VMs to Nodes
        for vm in proxmox.nodes(node["node"]).qemu.get():

            current_vm = VM(vm["vmid"])

            # TODO: Whether to only check running VMs?
            if vm["status"] == "stopped":
                continue

            vm_usage_data = proxmox.nodes(node["node"]).qemu(vm['vmid']).rrddata.get(timeframe="week", cf="AVERAGE")
            for data_entry in vm_usage_data:
                try:
                    current_vm.utilization.append(
                        VmUsageRecord(
                            data_entry["time"],
                            data_entry["disk"],
                            data_entry["maxdisk"],
                            data_entry["cpu"],
                            data_entry["maxcpu"],
                            data_entry["netin"],
                            data_entry["netout"],
                            data_entry["diskread"],
                            data_entry["diskwrite"],
                            data_entry["mem"],
                            data_entry["maxmem"]
                        )
                    )
                except:
                    # Data entry doesn't fit schema
                    print("Data entry doesn't fit the schema for:\n{0}".format(vm["name"]))
                    for key in data_entry.keys():
                        print("\t{0}: {1}".format(key,data_entry[key]))
            # Add VM to corresponding node
            current_node.vms.append(current_vm)
        cluster.nodes.append(deepcopy(current_node))
        del current_node
    return cluster
def plot_cluster_cpu_usage(cluster):
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    cluster.nodes[0].plot(fig, ax1, 1)
    cluster.nodes[1].plot(fig, ax2, 1)
    cluster.nodes[2].plot(fig, ax3, 1)
    plt.show()
def main():
    start_time = time.time()
    cluster = load_cluster_info()
    print("Cluster information loaded in --- %s seconds ---" % round(time.time() - start_time,3))

    plot_cluster_cpu_usage(cluster)
if __name__ == "__main__":
    main()

