from datetime import datetime
from math import floor, fsum
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.ticker as mtick

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
    
    def get_cpu_utilization_list(self):
        result = []
        for record in self.utilization:
            result.append(record.cpu)
        return result

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

    def get_vm_by_vmid(self, vmid):
        for index, vm in enumerate(self.vms):
            if str(vm.id) == str(vmid):
                return index, vm
        raise Exception("There is no vm with id {0} on {1} node".format(vmid, self.name))


    # Recalculates aggregate cpu utilization of all vms
    def calculate_aggregate_utilization(self):
        aggregate_utilization = []
        for i in range(0, len(self.utilization)):
            cpu_utilization_sum = 0
            for vm in self.vms:
                try:
                    if vm.utilization[i].time != self.utilization[i].time:
                        _, current_vm_utilization_record = vm.get_utilization_by_timestamp(self.utilization[i].time)
                    else:
                        current_vm_utilization_record = vm.utilization[i]
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
    
    def get_cluster_score(self, threshold = 0.15, sla_threshold = 0.85):
        total_sum = 0
        n = 0
        active_n = 0
        sla_n = 0
        for i in range(0, len(self.nodes)):
            for record in self.nodes[i].aggregate_utilization:
                    if record >= threshold:
                        total_sum += record
                        n += 1
                    if record >= sla_threshold:
                        sla_n += 1
        return (sla_n/n, sla_n)
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
