from proxmoxer import ProxmoxAPI
import yaml


class VmUsageRecord:
    maxcpu = 0
    netout = 0
    netin  = 0
    cpu    = 0
    diskread = 0
    diskwrite = 0
    mem = 0
    maxdisk = 0
    maxmem = 0
    time = 0
    disk = 0

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
    maxcpu = 0
    netout = 0
    netin  = 0
    cpu    = 0
    swapused = 0
    swaptotal = 0
    memtotal = 0
    memused = 0
    time = 0
    loadavg = 0
    iowait = 0
    rootused = 0

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
    id = 0
    utilization = []

    def __init__(self, id, utilization = []):
        self.id = id
        self.utilization = utilization

    def add_usage_record(record:VmUsageRecord):
        self.utilization.add(record)

class Node:
    name = ""
    vms = []
    utilization = []

    def __init__(self, name = "", vms = [], utilization = []):
        self.name = name
        self.vms = vms
        self.utilization = utilization
    
    def add_usage_record(record:NodeUsageRecord):
        self.utilization.add(record)



class Cluster:
    def __init__(self, name, nodes = []):
        self.name = name
        self.nodes = nodes

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

    for node in proxmox.nodes.get():
        
        current_node = Node(node["node"])

        # Gather and add node utilization history
        node_usage_data = proxmox.nodes(node["node"]).rrddata.get(timeframe="week", cf="MAX")
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

            vm_usage_data = proxmox.nodes(node["node"]).qemu(vm['vmid']).rrddata.get(timeframe="week", cf="AVERAGE")
            for data_entry in vm_usage_data:
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
            # Add VM to corresponding node
            current_node.vms.append(current_vm)
        cluster.nodes.append(current_node)
    return cluster

def main():
    cluster = load_cluster_info()

if __name__ == "__main__":
    main()
