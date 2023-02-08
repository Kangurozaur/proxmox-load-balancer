from copy import deepcopy
import time
from balancer.model import Cluster
from .simulation import migrate
from .util import Config
from .data_interface import perform_migration
import logging


def find_migrations(cluster: Cluster):
    migration_dict = {}
    for time_index in range(0, len(cluster.nodes[0].aggregate_utilization)):
        migrations = mm_policy(cluster.nodes, time_index)
        for host_index, migration_list in enumerate(migrations):
            for migration in migration_list:
                _, existing_count = migration_dict.get(migration, (0, 0))
                migration_dict[migration] = (host_index, existing_count + 1)
    return migration_dict


def mm_policy(host_list, time_index):
    
    THRESH_UP = 0.85
    migration_list = [[] for x in range(len(host_list))]

    # MM algorithm
    for host_index, host in enumerate(host_list):
        vm_list = []
        # Prepare vm list
        for vm in host.vms:
            try:
                vm_list.append((vm.id, vm.utilization[time_index].cpu * vm.utilization[time_index].maxcpu))
            except:
                pass
        vm_list.sort(key=lambda a: a[1], reverse = True)

        h_max_cpu = host.utilization[time_index].maxcpu
        h_util = host.aggregate_utilization[time_index] * h_max_cpu 
        thresh_up_cores = THRESH_UP * h_max_cpu
        best_fit_util = 100
        best_fit_vm = None
        while h_util > thresh_up_cores:
            # SLA violation occurs
            for vm_index, (vm_id, vm_util) in enumerate(vm_list):
                if vm_util > h_util - thresh_up_cores:
                    t = vm_util - h_util + thresh_up_cores
                    if t < best_fit_util:
                        best_fit_vm = vm_index
                else:
                    if best_fit_util == 100:
                        best_fit_vm = vm_index
                    break
            vm_id, best_fit_util = vm_list[best_fit_vm]
            h_util -= best_fit_util
            migration_list[host_index].append(vm_id)
            del(vm_list[best_fit_vm])
            best_fit_util = 100
            best_fit_vm = None
    return migration_list

def get_migration_cost(cluster, migration):
    config = Config.getInstance().config
    bandwidth = config["parameters"]["bandwidth"]
    _, vm = cluster.nodes[migration[1][0]].get_vm_by_vmid(migration[0])
    time_interval = vm.utilization[1].time - vm.utilization[0].time
    return (vm.utilization[0].disk + vm.utilization[0].mem)/(1000000*(bandwidth/8) * time_interval)

def balance_cluster(cluster):
    performed_migrations = []
    final_scores = []
    config = Config.getInstance().config
    w_sla = config["parameters"]["weight_sla"]
    w_mig = config["parameters"]["weight_mig"]
    vm_selection_depth = config["parameters"]["vm_selection_depth"]

    i = 0
    while True:
        start_time = time.time()
        migrations = find_migrations(cluster)
        _, old_score = cluster.get_cluster_score()
        current_migration = None
        allocations = []
        migrationsList = [(k, v) for k, v in migrations.items()]
        migrationsList.sort(key=lambda a: a[1][1], reverse=False)
        for j in range(0, vm_selection_depth):
            key = migrationsList[j][0]
            current_migration = (key, migrations[key])
        
            # Perform the current migration for each potential target node
            scores = [(100000000, 1000000000, 100000000) for x in range(0, len(cluster.nodes))]
            clusters = []
            for j in range(0, len(cluster.nodes)):
                new_cluster = None
                if j != current_migration[1][0]:
                    new_cluster = deepcopy(cluster)
                    migrate(current_migration[0], new_cluster.nodes[current_migration[1][0]], new_cluster.nodes[j], new_cluster)
                    (sla_score, sla_count) = new_cluster.get_cluster_score()
                    mig_cost = get_migration_cost(cluster, current_migration)
                    scores[j] = (w_sla * sla_score  + w_mig * mig_cost, sla_count, mig_cost) 
                clusters.append(new_cluster)
            #print(scores)
            # Select the best migration
            min_score = old_score
            best_cluster = cluster
            temp_perf_migrations = None
            for index, (sla_count, score, mig_time) in enumerate(scores):
                if score < min_score:
                    min_score = score
                    best_cluster = clusters[index]
                    temp_perf_migrations = deepcopy(performed_migrations)
                    temp_perf_migrations.append({"vm_id":current_migration[0], "source_node": clusters[index].nodes[current_migration[1][0]].name, "target_node": clusters[index].nodes[index].name, "old_score": old_score, "new_score": min_score, "sla_count": sla_count, "mig_time":mig_time})
            allocations.append((min_score, best_cluster, temp_perf_migrations))
        
        # Select the best of all migrations performed
        min_score = old_score
        for j in range(0, len(allocations)):
            temp_score, temp_cluster, temp_perf_migrations = allocations[j]
            if (temp_score < min_score):
                best_cluster = temp_cluster
                min_score = temp_score
                performed_migrations = temp_perf_migrations
        cluster = deepcopy(best_cluster)

        print("Iteration: {0} Cost decrease: {1}%. Current cost: {2}.".format(i, round((old_score-min_score)/old_score * 100, 2), min_score))
        logging.info("Iteration: {0} Cost decrease: {1}%. Current cost: {2}.".format(i, round((old_score-min_score)/old_score * 100, 2), min_score))
        print("Iteration performed in --- %s seconds ---" % round(time.time() - start_time,3))
        logging.info("Iteration performed in --- %s seconds ---" % round(time.time() - start_time,3))
        
        final_scores.append(min_score)
        i += 1
        if (old_score == min_score):
            break
    for migration in performed_migrations:
        perform_migration(migration)
    return cluster