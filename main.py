from copy import deepcopy
from datetime import datetime
from math import floor, fsum
from tokenize import Number
from proxmoxer import ProxmoxAPI
import yaml
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.ticker as mtick
import numpy as np
import random as rand

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
    def __str__(self):
        return "time: {0},\ndisk: {1},\nmaxdisk: {2},\ncpu: {3},\nmaxcpu: {4},\nnetin:{5},\nnetout:{6},\ndiskread:{7},\ndiskwrite:{8},\nmem:{9},\nmaxmem:{10}".format(self.time, self.disk, self.maxdisk, self.cpu, self.maxcpu, self.netin, self.netout, self.diskread, self.diskwrite, self.mem, self.maxmem)

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

    def __str__(self):
        return "time: {0},\nmaxcpu: {1},\nnetout: {2},\nnetin: {3},\ncpu: {4}, \nswapused:{5}, \nswaptotal:{6}, \nmemtotal:{7}, \nmemused:{8}, \nloadavg:{9},\niowait:{10},\nrootused:{11},\nroottotal:{12}".format(self.time, self.maxcpu, self.netout, self.netin, self.cpu, self.swapused, self.swaptotal, self.memtotal, self.memused, self.loadavg, self.iowait, self.rootused, self.roottotal)


class VM:

    def __init__(self, id):
        self.id = id
        self.utilization = []

    def add_usage_record(record:VmUsageRecord):
        self.utilization.add(record)

    def get_utilization_by_timestamp(self, timestamp):
        for index, record in enumerate(self.utilization):
            if record.time == timestamp:
                return index, record
        raise Exception("There are no utilization records for {0} timestamp for vm {1}".format(timestamp, self.id))

class Node:

    def __init__(self, name = ""):
        self.name = name
        self.vms = []
        self.utilization = []
        self.aggregate_utilization = []
        self.vcpu_ratio = 4
    
    def add_usage_record(self, record:NodeUsageRecord):
        self.utilization.add(record)

    def add_vm(self, vm):
        self.vms.append(vm)
    
    def get_utilization_by_timestamp(self, timestamp) -> NodeUsageRecord:
        for index, record in enumerate(self.utilization):
            if timestamp == record.time:
                return index, record
        raise Exception("There are no utilization records for {0} timestamp for node {1}".format(timestamp, self.name))

    def get_vm_by_vmid(self, vmid):
        for index, vm in enumerate(self.vms):
            if vm.id == vmid:
                return index, vm
        raise Exception("There are is no vm with id {0} on {1} node".format(vmid, self.name))

    # Recalculates aggregate cpu utilization of all vms
    def calculate_aggregate_utilization(self):
        aggregate_utilization = []
        for i in range(0, len(self.utilization)):
            cpu_utilization_sum = 0
            for vm in self.vms:
                _, current_vm_utilization_record = vm.get_utilization_by_timestamp(self.utilization[i].time)
                # Include number of cores in calculation
                cpu_utilization_sum += current_vm_utilization_record.cpu * current_vm_utilization_record.maxcpu
            aggregate_utilization.append(cpu_utilization_sum)
        self.aggregate_utilization = aggregate_utilization

    def plot(self, fig, ax, n_aggregate=1):
        # Plot Node usage records
        # Works under assumption - all corresponding records share same timestamps
        n_time_points = len(self.aggregate_utilization)
        xpoints = []
        ypoints = []

        for i in range(0, floor(n_time_points/n_aggregate)):
            time_sum = 0.0
            sum = 0.0

            for j in range(0, n_aggregate):
                time_sum += self.utilization[i*n_aggregate + j].time
                sum += self.aggregate_utilization[i*n_aggregate + j]
            
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
        # Show 100%
        ax.axhline(y = 1, color = 'r', linestyle = '-')
        # Show average across whole period
        ax.axhline(y = fsum(ypoints)/len(ypoints), color = 'y', linestyle = '-')
        return (xpoints, ypoints)

class Cluster:
    def __init__(self, name):
        self.name = name
        self.nodes = []
    
    def get_average_cpu_usage(self):
        # For each node get total usage per timestamp
        total_sum = 0
        for i in range(0, len(self.nodes)):
            self.nodes[i].calculate_aggregate_utilization()
            for record in self.nodes[i].aggregate_utilization:
            #    if record > 1.0:
            #        total_sum += 1
            #    else:
                    total_sum += record
        return total_sum/(len(self.nodes[0].aggregate_utilization * len(self.nodes)))

# Pulls cluster state for a given timeframe from ProxMox API
# returns a Cluster state object
def load_cluster_info() -> Cluster:

    cluster = Cluster("cluster_1")

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
    n_nodes = len(cluster.nodes)
    fig, ax = plt.subplots(n_nodes+1,1)

    x_points = []
    y_sum = []
    y_sum_ceiling = []
    #TODO: Code cleanup
    for i in range(0, n_nodes):
        cluster.nodes[i].calculate_aggregate_utilization()
        # Calculate aggregated CPU utilization
        if i==0:
            (xpoints,y_sum) = cluster.nodes[i].plot(fig, ax[i],1)
            
            y_temp = deepcopy(y_sum)
            # Ceiling of 1.0 for each element
            #for j in range(0, len(y_temp)):
            #    if y_temp[j] > 1.0:
            #        y_temp[j] = 1.0
            y_sum_ceiling = y_temp
        else:
            (_ ,y_temp) = cluster.nodes[i].plot(fig, ax[i],1)
            y_sum = list(map(lambda a,b: a + b, y_sum, y_temp))

            # Ceiling of 1.0 for each element
            #for j in range(0, len(y_temp)):
            #    if y_temp[j] > 1.0:
            #        y_temp[j] = 1.0
            y_sum_ceiling = list(map(lambda a,b: a + b, y_sum_ceiling, y_temp))
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
    plt.show()

def migrate(vmid, source_host: Node, target_host: Node):
    # Move vm from source to host

    # Modify utilization records accordingly
    vm_index, vm = source_host.get_vm_by_vmid(vmid)

    target_host.add_vm(deepcopy(vm))
    del source_host.vms[vm_index]

    #print("Migrated vm: {0}\n\tfrom: {1}\n\tto: {2}".format(vmid, source_host.name, target_host.name))
    
    #for count, record in enumerate(vm.utilization):
        # Reduce usage on source host
        #index, node_record = source_host.get_utilization_by_timestamp(record.time)
        #print("Count: {0}, timestamp: {1}\n\t".format(count, record.time))
        #print(record)
        #print(node_record)

def perform_migrations(cluster: Cluster):

    # Find time at which aggregated utilization is over 100%
    # for index, utilization_record in enumerate(cluster.nodes[0].aggregate_utilization):
    #     if utilization_record > 1.5:
    #         # Find which hosts are underloaded at that time
    #         min_load = 100000.0
    #         migration_target = 0
    #         # For each node, except first
    #         for i in range(1, len(cluster.nodes)):
    #             if cluster.nodes[i].aggregate_utilization[index] < min_load:
    #                 min_load = cluster.nodes[i].aggregate_utilization[index]
    #                 migration_target = i
    #         # If no candidate found, move on
    #         if migration_target != 0:
    #             # Migrate vms until capacity is reached
    #             j = 0
    #             while cluster.nodes[0].aggregate_utilization[index] >= 1.0 and min_load < 1.0:
    #                 migrate(cluster.nodes[0].vms[j].id, cluster.nodes[0], cluster.nodes[migration_target])
    #                 cluster.nodes[0].calculate_aggregate_utilization()
    #                 cluster.nodes[migration_target].calculate_aggregate_utilization()
    #                 min_load = cluster.nodes[migration_target].aggregate_utilization[index]
    #                 j += 1
    return cluster
                
def random_migration(cluster: Cluster) -> Cluster:
    # Create cluster copy
    cluster = deepcopy(cluster)
    source_host_index = 0
    target_host_index = 0

    # Select random nodes until they are not the same
    while source_host_index == target_host_index:
        source_host_index = rand.randint(0, len(cluster.nodes)-1)
        target_host_index = rand.randint(0, len(cluster.nodes)-1)

    # Assign hosts to variables by index
    source_host = cluster.nodes[source_host_index]
    target_host = cluster.nodes[target_host_index]

    # Select random vm for migration
    vm_index = rand.randint(0,len(source_host.vms)-1)
    vmid = source_host.vms[vm_index].id
    
    # Migrate it
    migrate(vmid, source_host, target_host)
    return cluster

def main():
    start_time = time.time()
    cluster = load_cluster_info()
    print("Cluster information loaded in --- %s seconds ---" % round(time.time() - start_time,3))

    #for vm in cluster.nodes[0].vms:
        #for record in vm.utilization:
            #print(record.cpu)
            #print(record.maxcpu)

    score = cluster.get_average_cpu_usage()
    print(score)
    
    start_time = time.time()
    for i in range(0, 100000):
        new_state = random_migration(cluster)
        new_state_score = new_state.get_average_cpu_usage()
        if  new_state_score > score:
            score = new_state_score
            cluster = new_state
    #plot_cluster_cpu_usage(cluster)
    print("Migrations performed in --- %s seconds ---" % round(time.time() - start_time,3))
    print(cluster.get_average_cpu_usage())

    plot_cluster_cpu_usage(cluster)

if __name__ == "__main__":
    main()

