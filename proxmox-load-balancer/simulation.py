from copy import deepcopy
from model import Node

def round_robin_scheduler(node, index, core_coef = 0.001, vm_id = None):
    # Initialize an empty list to store the scheduled cores for each virtual machine
    scheduled_cores = [0 for vm in node.vms]

    # Calculate the total number of virtual cores assigned to the virtual machines on the node
    total_vm_cores = sum([vm.utilization[index].maxcpu for vm in node.vms])

    idle_constants = [0.05 for vm in node.vms]

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
                        if node.vms[j].utilization[next_index].cpu > 2*idle_constants[j]:
                            # VM not idle at that point
                            # Workload left incremented, but excluding idle state
                            workload_left += (node.vms[j].utilization[next_index].maxcpu * node.vms[j].utilization[next_index].cpu) - (idle_constants[j] * node.vms[j].utilization[next_index].maxcpu)
                        else:
                            break                   
                    if workload_left - (scheduled_cores[j] - old_utilization) > 0:
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
def reschedule_node(node, overload_timestamps, vm_id = None):
    
    j = 0
    while j < len(overload_timestamps):
        time_index = overload_timestamps[j]
        scheduled_cores, idle_constants = round_robin_scheduler(node, time_index, 0.01, vm_id)
        
        # Reassign the newly scheduled cores for each VM, along with the utilization
        for i, new_cores in enumerate(scheduled_cores):
            old_cores = node.vms[i].utilization[time_index].cpu * node.vms[i].utilization[time_index].maxcpu
            additional_workload = new_cores - old_cores
            # If additional workload positive - take workload from somewhere
            if additional_workload > 0.01:
                # Find where the current workload ends
                workload_end_index = len(node.vms[i].utilization) - 1
                for next_index in range(time_index, len(node.vms[i].utilization)):
                    if node.vms[i].utilization[next_index].cpu * node.vms[i].utilization[next_index].maxcpu <= idle_constants[i]:
                        workload_end_index = next_index
                        break
                for next_index in reversed(range(time_index+1, workload_end_index)):
                    available_workload = node.vms[i].utilization[next_index].cpu * node.vms[i].utilization[next_index].maxcpu - idle_constants[i] * node.vms[i].utilization[next_index].maxcpu
                    if available_workload < 0:
                        continue
                    if additional_workload > available_workload:
                        additional_workload -= available_workload
                        node.vms[i].utilization[next_index].cpu = idle_constants[i]
                    else:
                        node.vms[i].utilization[next_index].cpu -= additional_workload / node.vms[i].utilization[next_index].maxcpu
                        break 
                node.vms[i].utilization[time_index].cpu = new_cores / node.vms[i].utilization[time_index].maxcpu
            elif additional_workload < 0:
                # If additional workload negative - delegate it further
                node.vms[i].utilization[time_index+1].cpu -= additional_workload / node.vms[i].utilization[time_index+1].maxcpu
                node.vms[i].utilization[time_index].cpu = new_cores / node.vms[i].utilization[time_index].maxcpu
                # Add next timestamp for rescheduling if its not there already
                if j == len(overload_timestamps) - 1 or overload_timestamps[j+1] != time_index + 1:
                    overload_timestamps.insert(j+1, time_index+1)
        j += 1

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