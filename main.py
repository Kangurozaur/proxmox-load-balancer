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
from prometheus_api_client.utils import parse_datetime
from datetime import timedelta

class VmUsageRecord:

    def __init__(self, time, cpu, maxcpu, disk = 0, maxdisk = 0, netin = 0, netout = 0, diskread = 0, diskwrite = 0, mem = 0, maxmem = 0):
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

    def __init__(self, time, cpu, maxcpu, netout = 0, netin = 0, swapused = 0, swaptotal = 0, memtotal = 0, memused = 0, loadavg = 0, iowait = 0, rootused = 0, roottotal = 0):
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

    def add_usage_record(self, record:VmUsageRecord):
        self.utilization.append(record)

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
        self.utilization.append(record)

    # Add VM to a node. If it already exists, include missing usage records
    def add_vm(self, vm):
        try:
            _, existing_vm = self.get_vm_by_vmid(vm.id)
            for index, utilization_record in enumerate(existing_vm.utilization):
                if utilization_record.cpu == 0:
                    existing_vm.utilization[index] = vm.utilization[index]
        except:
            self.vms.append(vm)
    
    def get_utilization_by_timestamp(self, timestamp) -> NodeUsageRecord:
        for index, record in enumerate(self.utilization):
            if timestamp == record.time:
                return index, record
        return NodeUsageRecord(timestamp,0,0)
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
                except Exception as e:
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
            xpoints.append(mdate.date2num(datetime.utcfromtimestamp((time_sum/n_aggregate))))

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
                        current_vm.utilization.append(VmUsageRecord(data_entry["time"], 0, 0))
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

            migration_timestamp_index = len(target_vm.utilization)       # Index of record when migration occured, 0-69
            for index, utilization_record in enumerate(target_vm.utilization):
                if utilization_record.time < migration["endtime"]:
                    # Remove utilization records from before migration
                    target_vm.utilization[index].cpu = 0
                else:
                    migration_timestamp_index = index
                    break
            for index in range(migration_timestamp_index, len(current_vm.utilization)):
                current_vm.utilization[index].cpu = 0
            source_node.add_vm(current_vm)
                  
    elif config["method"] == "prometheus":
        prom = PrometheusConnect(url=config["connection"]["prometheus"]["url"], disable_ssl=True)
        
        start_time = parse_datetime("7d")
        end_time = parse_datetime("now")
        chunk_size = 60.0

        cpu_usage_data = prom.custom_query_range(
            '(pve_cpu_usage_ratio * on(id, instance) group_left(name, type, node) pve_guest_info{instance = "10.71.172.10", type = "qemu"}) and on(id, instance) pve_up == 1',
            start_time=start_time,
            end_time=end_time,
            step=chunk_size
        )

        cpu_max_data = prom.custom_query_range(
            '(pve_cpu_usage_limit * on(id, instance) group_left(name, type,node) pve_guest_info{instance = "10.71.172.10", type = "qemu"}) and on(id, instance) pve_up == 1',
            start_time=start_time,
            end_time=end_time,
            step=chunk_size
        )

        nodes_cpu_max_data = prom.custom_query_range(
            'pve_cpu_usage_limit{instance = "10.71.172.10", id =~"node/.*"} and on(id, instance) pve_up == 1',
            start_time=start_time,
            end_time=end_time,
            step=chunk_size
        )

        nodes_cpu_usage_data = prom.custom_query_range(
            '(pve_cpu_usage_ratio * on(id, instance) pve_cpu_usage_limit{instance = "10.71.172.10", id=~"node.*"}) and on(id, instance) pve_up == 1',
            start_time=start_time,
            end_time=end_time,
            step=chunk_size
        )

        # Add all nodes

        for i in range(0, len(nodes_cpu_max_data)):

            current_node = Node(nodes_cpu_max_data[i]["metric"]["id"].split("/")[1])
            
            for node_series_index, node_series in enumerate(nodes_cpu_usage_data):
                if node_series["metric"]["id"].split("/")[1] == current_node.name:
                    for series_index, cpu_usage_series in enumerate(node_series["values"]):
                        current_usage_record = NodeUsageRecord(time=cpu_usage_series[0], cpu=float(cpu_usage_series[1]), maxcpu=int(nodes_cpu_max_data[node_series_index]["values"][series_index][1]))
                        current_node.add_usage_record(current_usage_record)
                    break
            cluster.nodes.append(current_node)

        for entry_index, cpu_data_entry in enumerate(cpu_usage_data):

            current_node = cluster.get_node_by_name(cpu_data_entry["metric"]["node"])

            vm_id = cpu_data_entry["metric"]["id"].split("/")[1]
            
            current_vm = None
            try:
                current_vm = current_node.get_vm_by_vmid(vm_id)
            except:
                current_vm = VM(vm_id)
                current_node.add_vm(current_vm)
            
            for series_index, series in enumerate(cpu_data_entry["values"]):
                current_usage_record = VmUsageRecord(time=series[0], cpu=float(series[1]), maxcpu=int(cpu_max_data[entry_index]["values"][series_index][1]))
                current_vm.add_usage_record(current_usage_record)
            
        _ = cluster.get_average_cpu_usage()

    return cluster

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

def round_robin_scheduler(node, index, core_coef = 0.001, vm_id = None):
    # Initialize an empty list to store the scheduled cores for each virtual machine
    scheduled_cores = [0 for vm in node.vms]

    # Calculate the total number of virtual cores assigned to the virtual machines on the node
    total_vm_cores = sum([vm.utilization[index].maxcpu for vm in node.vms])

    idle_constants = [min(record.cpu for record in vm.utilization) for vm in node.vms]

    node_cores = node.utilization[index].maxcpu
    # Check if the node is overcommitted
    if total_vm_cores > node_cores:
        # Calculate the number of times the allocation process should repeat
        repeat = node_cores * int(1/core_coef)

        # Allocate the cores to the virtual machines in a round-robin fashion
        was_core_assigned = True
        while repeat != 0 and was_core_assigned:
            was_core_assigned = False
            for j in range(len(node.vms)):
                # If no cores left to assign
                if repeat == 0:
                    break
                old_utilization = node.vms[j].utilization[index].maxcpu * node.vms[j].utilization[index].cpu
                next_utilization = node.vms[j].utilization[index+1].cpu * node.vms[j].utilization[index+1].maxcpu
                # Check if the current virtual machine is allocated max number of cores
                if scheduled_cores[j] >= node.vms[j].utilization[index].maxcpu:
                    scheduled_cores[j] = node.vms[j].utilization[index].maxcpu
                    continue
                elif scheduled_cores[j] < old_utilization:
                    scheduled_cores[j] += core_coef
                    repeat -= 1
                    was_core_assigned = True
                # Check if additional cores should be allocated - this is done when a big workload is ran continuously
                # and there is enough workload to "take" from next timestamps, until one with idle load is found
                elif vm_id == None or node.vms[j].id == vm_id:
                    workload_left = 0
                    for next_index in range(index, len(node.vms[j].utilization)):
                        if node.vms[j].utilization[next_index].cpu > idle_constants[j]*2:
                            # VM not idle at that point
                            # Workload left incremented, but excluding idle state
                            workload_left += (node.vms[j].utilization[next_index].maxcpu * node.vms[j].utilization[next_index].cpu) - (idle_constants[j] * node.vms[j].utilization[next_index].maxcpu)
                        else:
                            break                   
                    if workload_left - (scheduled_cores[j] - old_utilization) > idle_constants[j]:
                        scheduled_cores[j] += core_coef
                        repeat -= 1
                        was_core_assigned = True

    else:
        # The node is not overcommitted, so allocate the required number of cores to each virtual machine
        for j in range(len(node.vms)):
            scheduled_cores[j] = node.vms[j].utilization[index].maxcpu * node.vms[j].utilization[index].cpu

    return scheduled_cores, idle_constants

# Function that does ...
# vm_id - id of migrated vm; if the rescheduling is done on a target node after migration and target node is not overcommited or None otherwise
def reschedule_node(node, overload_timestamps, vm_id = None, idle_threshold = 100):
    for time_index in overload_timestamps:
        scheduled_cores, idle_constants = round_robin_scheduler(node, time_index, 0.01, vm_id)
        
        # Reassign the newly scheduled cores for each VM, along with the utilization
        for i, new_cores in enumerate(scheduled_cores):
            old_cores = node.vms[i].utilization[time_index].cpu * node.vms[i].utilization[time_index].maxcpu
            additional_workload = new_cores - old_cores
            # If additional workload positive - take workload from somewhere
            if additional_workload > 0:
                # Find where the current workload ends
                workload_end_index = len(node.vms[i].utilization) - 1
                for next_index in range(time_index, len(node.vms[i].utilization)):
                    if node.vms[i].utilization[next_index].cpu * node.vms[i].utilization[next_index].maxcpu < idle_threshold * idle_constants[i]:
                        workload_end_index = next_index
                for next_index in reversed(range(time_index+1, workload_end_index)):
                    if additional_workload > (idle_constants[i] * node.vms[i].utilization[next_index].maxcpu) + (node.vms[i].utilization[next_index].cpu * node.vms[i].utilization[next_index].maxcpu):
                        additional_workload -= node.vms[i].utilization[next_index].cpu * node.vms[i].utilization[next_index].maxcpu
                        node.vms[i].utilization[next_index].cpu = idle_constants[i]
                    else:
                        node.vms[i].utilization[next_index].cpu -= additional_workload / node.vms[i].utilization[next_index].maxcpu
                        break 
                node.vms[i].utilization[time_index].cpu = new_cores / node.vms[i].utilization[time_index].maxcpu
            elif additional_workload < 0:
                # If additional workload negative - delegate it further
                node.vms[i].utilization[time_index+1].cpu -= additional_workload / node.vms[i].utilization[time_index+1].maxcpu
                node.vms[i].utilization[time_index].cpu = new_cores / node.vms[i].utilization[time_index].maxcpu

def get_overload_timestamps(node, overload_threshold = 0.95):
    # Find at what times was the source node overcommited
    overload_timestamps = []
    for index, record in enumerate(node.aggregate_utilization):
        if record >= overload_threshold:
            # It was overloaded
            overload_timestamps.append(index)
    return overload_timestamps

def migrate(vmid, source_host: Node, target_host: Node, cluster, max_depth = 100):
    # Move vm from source to host

    vm_index, vm = source_host.get_vm_by_vmid(vmid)

    current_vm = deepcopy(vm)

    # Modify utilization records accordingly 

    # Adjust cpu usage on source node
    _ = cluster.get_average_cpu_usage()

    overload_timestamps = get_overload_timestamps(source_host, 0.95)
    # Delete the migrated VM
    del source_host.vms[vm_index]
    # For each of these timestamps, reschedule cores
    
    reschedule_node(source_host, overload_timestamps)

    source_host.calculate_aggregate_utilization()

    # Repeat until all overload cases solved or depth exceeded
    counter = 0
    while True and counter < max_depth:
        temp_overload_timestamps = get_overload_timestamps(source_host, 1.0)
        if (len(overload_timestamps) == 0):
            break
        else:
            reschedule_node(source_host, temp_overload_timestamps)
            source_host.calculate_aggregate_utilization()
            counter += 1


    # Adjust cpu usage on target node

    target_host.add_vm(current_vm)

    # Recalculate summed up utilization to see if target node gets overcommited at any point due to migration
    _ = cluster.get_average_cpu_usage()

    # Also, reschedule cores for timestamps when the source node was overloaded (even if target host doesn't)

    if (len(get_overload_timestamps(target_host)) == 0):
        reschedule_node(target_host, overload_timestamps, vmid)
        target_host.calculate_aggregate_utilization()

    overload_timestamps = get_overload_timestamps(target_host)

    reschedule_node(target_host, overload_timestamps)
    target_host.calculate_aggregate_utilization()
    
    # Repeat until all overload cases solved or depth exceeded
    counter = 0
    while True and counter < max_depth:
        temp_overload_timestamps = get_overload_timestamps(target_host, 1.0)
        if (len(temp_overload_timestamps) == 0):
            break
        else:
            reschedule_node(target_host, temp_overload_timestamps)
            target_host.calculate_aggregate_utilization()
            counter += 1

                
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
    migrate(vmid, source_host, target_host, cluster)
    return cluster

def load_cluster_from_file(filename):
    with open(filename, 'rb') as cluster_file:
        cluster = pickle.load(cluster_file)
    return cluster

def save_cluster_to_file(cluster, filename):
    with open(filename, 'wb') as cluster_snapshot:
        pickle.dump(cluster, cluster_snapshot)

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

def print_results(p_coefficients):
    # Calculate number of scenarios
    num_scenarios = len(p_coefficients) // 2

    # Print header row
    print("Scenario\tResult")
    print("--------\t------")

    # Print results for each scenario
    for i in range(num_scenarios):
        # Calculate average Pearson correlation coefficient for scenario
        avg = (p_coefficients[2 * i] + p_coefficients[2 * i + 1]) / 2

        # Print results for scenario
        print("%d\t\t%.4f" % (i + 1, avg))

def test_simulation():
    simulation_timeframe = [(49,68),(42,65),(43,70)]
    reference_timeframe = [(50,69),(44,67),(22,49)]

    p_coefficients = []
    for n_scenario in range(1,4):
        base_state = load_cluster_from_file('snapshots/scenario{0}/snapshot-before-hour-AVERAGE'.format(n_scenario))

        expected_state = load_cluster_from_file('snapshots/scenario{0}/snapshot-after-hour-AVERAGE'.format(n_scenario))

        simulated_state = deepcopy(base_state)
        
        # Trim cluster utilization entries to period of interest
        simulated_state = trim_cluster_data(simulated_state, simulation_timeframe[n_scenario-1])
        expected_state = trim_cluster_data(expected_state, reference_timeframe[n_scenario-1])

        migrate(114, simulated_state.get_node_by_name("pve-g2n6"), simulated_state.get_node_by_name("pve-g2n8"), simulated_state)

        plot_cluster_cpu_usage(simulated_state, ["pve-g2n6", "pve-g2n8"])
        plot_cluster_cpu_usage(expected_state, ["pve-g2n6", "pve-g2n8"])

        _ = simulated_state.get_average_cpu_usage()
        _ = expected_state.get_average_cpu_usage()

        utilization_reference_node1 = expected_state.get_node_by_name("pve-g2n6").aggregate_utilization
        utilization_expected_node1 = simulated_state.get_node_by_name("pve-g2n6").aggregate_utilization

        utilization_reference_node2 = expected_state.get_node_by_name("pve-g2n8").aggregate_utilization
        utilization_expected_node2 = simulated_state.get_node_by_name("pve-g2n8").aggregate_utilization

        r1 = pearson_coefficient(utilization_reference_node1, utilization_expected_node1)
        r2 = pearson_coefficient(utilization_reference_node2, utilization_expected_node2)

        p_coefficients.append(r1)
        p_coefficients.append(r2)

    print_results(p_coefficients)

def main():

    #test_simulation()

    start_time = time.time()
    #cluster = load_cluster_info("week", "average")
    cluster = load_cluster_from_file("snapshots/data_week")
    
    #_ = cluster.get_average_cpu_usage()
    plot_cluster_cpu_usage(cluster)

    print("Cluster data loaded in --- %s seconds ---" % round(time.time() - start_time,3))


    # Plot all figures
    plt.show()

if __name__ == "__main__":
    main()

