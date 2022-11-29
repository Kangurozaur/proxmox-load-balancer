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
import pickle
import re
from prometheus_api_client import PrometheusConnect

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
        self.migrations = []
    
    def add_usage_record(self, record:NodeUsageRecord):
        self.utilization.add(record)

    # Add VM to a node. If it already exists, include missing usage records
    def add_vm(self, vm):
        try:
            _, existing_vm = self.get_vm_by_vmid(vm.id)
            for index, utilization_record in existing_vm.utilization:
                if utilization_record.cpu == 0:
                    existing_vm.utilization[index] = vm.utilization[index]
        except:
            self.vms.append(vm)
    
    def get_utilization_by_timestamp(self, timestamp) -> NodeUsageRecord:
        for index, record in enumerate(self.utilization):
            if timestamp == record.time:
                return index, record
        return NodeUsageRecord(timestamp,0,0,0,0,0,0,0,0,0,0,0,0)
        raise Exception("There are no utilization records for {0} timestamp for node {1}".format(timestamp, self.name))

    def get_vm_by_vmid(self, vmid):
        for index, vm in enumerate(self.vms):
            if str(vm.id) == str(vmid):
                return index, vm
        raise Exception("There are is no vm with id {0} on {1} node".format(vmid, self.name))


    # Recalculates aggregate cpu utilization of all vms
    def calculate_aggregate_utilization(self):
        aggregate_utilization = []
        for i in range(0, len(self.utilization)):
            cpu_utilization_sum = 0
            for vm in self.vms:
                try:
                    _, current_vm_utilization_record = vm.get_utilization_by_timestamp(self.utilization[i].time)
                    # Include number of cores in calculation
                    cpu_utilization_sum += (current_vm_utilization_record.cpu * current_vm_utilization_record.maxcpu/self.utilization[i].maxcpu)
                except:
                    pass                
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
                    total_sum += record
        return total_sum/(len(self.nodes[0].aggregate_utilization * len(self.nodes)))
    
    def get_node_by_name(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        raise Exception("There are is no node with name {0} on {1} cluster".format(name, self.name))

    # Get names of all nodes
    def get_node_names(self):
        node_names = []
        for node in self.nodes:
            node_names.append(node.name)
        return node_names

# Pulls cluster state for a given timeframe from ProxMox API
# returns a Cluster state object
def load_cluster_info(timeframe="week", cf="AVERAGE") -> Cluster:

    cluster = Cluster("cluster_1")

    # Load config file
    config = yaml.safe_load(open("config.yaml"))

    # Initialize ProxmoxAPI
    if config["method"] == "proxmox":
        proxmox = ProxmoxAPI(config["connection"]["proxmox"]["url"]["ip"], user=config["connection"]["proxmox"]["auth"]["username"], password=config["connection"]["proxmox"]["auth"]["password"], verify_ssl=False)

        nodes_list = proxmox.nodes.get()
        for node in nodes_list:
            
            current_node = Node(node["node"])

            # Gather and add node utilization history
            node_usage_data = proxmox.nodes(node["node"]).rrddata.get(timeframe=timeframe, cf=cf)
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

                vm_usage_data = proxmox.nodes(node["node"]).qemu(vm['vmid']).rrddata.get(timeframe=timeframe, cf=cf)
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
                        print("Data entry doesn't fit the schema for:\n{0}\nAdding empty usage.".format(vm["name"]))
                        for key in data_entry.keys():
                            print("\t{0}: {1}".format(key,data_entry[key]))
                        current_vm.utilization.append(VmUsageRecord(
                            data_entry["time"],
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0
                        ))
                # Add VM to corresponding node
                current_node.vms.append(current_vm)

            # Read log for node to find and simulate any migrations performed
            # Requires Sys.audit permissions to display all logs
            reference_epoch = current_node.utilization[0].time
            logs = proxmox.nodes(node["node"]).tasks.get(typefilter="qmigrate", since=reference_epoch)
            for migration in logs:
                # Find target node
                log_text = proxmox.nodes(node["node"]).tasks(migration["upid"]).log.get()
                migration["target_node"] = re.findall(r"'(.+)'", log_text[0]["t"])[0]
            current_node.migrations = logs
            
            cluster.nodes.append(deepcopy(current_node))
        
        # For each migration, include the VM in both source and target nodes, but split utilization records depending on the migration timestamp
        # Migration format example:
        # {'endtime': 1669728729, 'status': 'OK', 'pstart': 1435759933, 'upid': 'UPID:pve-g2n6:00343D2D:5593F53D:638609B1:qmigrate:114:pibn@pve:', 'starttime': 1669728689, 'type': 'qmigrate', 'node': 'pve-g2n6', 'target_node': 'pve-g2n8', 'user': 'pibn@pve', 'pid': 3423533, 'id': '114'}
        
        initial_migrations = []
        for node in cluster.nodes:
            for migration in node.migrations:
                initial_migrations.append(migration)
        initial_migrations.sort(key=lambda a: a["endtime"], reverse=True)
        for migration in initial_migrations:
            # Initial migrations simulated from last to first
            source_node = cluster.get_node_by_name(migration["node"])
            target_node = cluster.get_node_by_name(migration["target_node"])
            _, target_vm = target_node.get_vm_by_vmid(migration["id"])
            
            # Check if VM records already exist on source node
            current_vm = deepcopy(target_vm)

            migration_timestamp_index = 0       # Index of record when migration occured, 0-69
            for index, utilization_record in enumerate(target_vm.utilization):
                if utilization_record.time < migration["endtime"]:
                    # Remove utilization records from before migration
                    target_vm.utilization[index].cpu = 0
                else:
                    migration_timestamp_index = index
                    continue
            for index in range(migration_timestamp_index, len(current_vm.utilization)):
                current_vm.utilization[index].cpu = 0
            source_node.add_vm(current_vm)
                  
    elif config.method == "prometheus":
        prom = PrometheusConnect(url ="<prometheus-host>", disable_ssl=True)
    return cluster

def plot_cluster_cpu_usage(cluster: Cluster, node_names_list = []):
    
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
            cluster.nodes[i].calculate_aggregate_utilization()
            # Calculate aggregated CPU utilization
            if plot_index == 0:
                (xpoints,y_sum) = cluster.nodes[i].plot(fig, ax[plot_index], 1)
                
                y_temp = deepcopy(y_sum)
                # Ceiling of 1.0 for each element
                #for j in range(0, len(y_temp)):
                #    if y_temp[j] > 1.0:
                #        y_temp[j] = 1.0
                y_sum_ceiling = y_temp
            else:
                (_ ,y_temp) = cluster.nodes[i].plot(fig, ax[plot_index], 1)
                y_sum = list(map(lambda a,b: a + b, y_sum, y_temp))

                # Ceiling of 1.0 for each element
                #for j in range(0, len(y_temp)):
                #    if y_temp[j] > 1.0:
                #        y_temp[j] = 1.0
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

def migrate(vmid, source_host: Node, target_host: Node):
    # Move vm from source to host

    vm_index, vm = source_host.get_vm_by_vmid(vmid)

    current_vm = deepcopy(vm)

    # Modify utilization records accordingly 

    # Normalize utilization to match target host CPUs
    cpu_factor = target_host.utilization[0].maxcpu/source_host.utilization[0].maxcpu
    for record in current_vm.utilization:
        record.cpu = record.cpu * cpu_factor

    # Adjust cpu usage on source node
    del source_host.vms[vm_index]
    

    # Adjust cpu usage on target node
    target_host.add_vm(current_vm)
    
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
    
    load_method = "API"
    timeframe = "day"
    save = False
    
    if load_method == "FILE":
        # Load from file
        with open('snapshots/scenario1/29112022145310-snapshot-before-hour', 'rb') as cluster_file:
            cluster = pickle.load(cluster_file)
    else:
        # Load from API
        cluster = load_cluster_info(timeframe,"AVERAGE")
    print("Cluster information loaded in --- %s seconds ---" % round(time.time() - start_time,3))
    
    # Save snapshot
    if save:
        with open("snapshots/scenario2/{0}-snapshot-before-{1}".format(datetime.today().strftime("%d%m%Y%H%M%S"), timeframe), 'wb') as cluster_snapshot:
            pickle.dump(cluster, cluster_snapshot)

    score = cluster.get_average_cpu_usage()
    start_score = score
    print(score)
    
    start_time = time.time()
    n_iterations = 0
    plot_cluster_cpu_usage(cluster, ["pve-g2n6", "pve-g2n8"])
    #migrate(114, cluster.get_node_by_name("pve-g2n6"), cluster.get_node_by_name("pve-g2n8"))
    #Recalculate score
    score = cluster.get_average_cpu_usage()
    #plot_cluster_cpu_usage(cluster)
    print("Migrations performed in --- %s seconds ---" % round(time.time() - start_time,3))
    print("Improvement from: {0} to {1}. (by {2}%)".format(start_score, score, (score-start_score)/start_score*100))

    # Case 1, test cluster:
    plot_cluster_cpu_usage(cluster, ["pve-g2n6", "pve-g2n8"])
    # Case 2, production cluster:
    #plot_cluster_cpu_usage(cluster)

    # Plot all figures
    plt.show()

if __name__ == "__main__":
    main()

