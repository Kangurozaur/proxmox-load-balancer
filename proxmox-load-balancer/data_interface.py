from copy import deepcopy
import pickle
from model import Cluster, VmUsageRecord, VM, Node, NodeUsageRecord
import yaml
import re
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime
from proxmoxer import ProxmoxAPI

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

def perform_migration(vmid, source_node_name, target_node_name):
    config = yaml.safe_load(open("config.yaml"))
    proxmox = ProxmoxAPI(config["connection"]["proxmox"]["url"]["ip"], user=config["connection"]["proxmox"]["auth"]["username"], password=config["connection"]["proxmox"]["auth"]["password"], verify_ssl=False)

    upid = proxmox.nodes(source_node_name).qemu(vmid).migrate.post(**{'target': target_node_name, 'online': 1, 'with-local-disks':1})

    return upid

def load_cluster_from_file(filename) -> Cluster:
    with open(filename, 'rb') as cluster_file:
        cluster = pickle.load(cluster_file)
    return cluster

def save_cluster_to_file(cluster, filename):
    with open(filename, 'wb') as cluster_snapshot:
        pickle.dump(cluster, cluster_snapshot)
    
