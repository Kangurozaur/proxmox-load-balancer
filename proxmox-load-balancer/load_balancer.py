from copy import deepcopy
from model import Cluster
from simulation import migrate


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
            vm_list.append((vm.id, vm.utilization[time_index].cpu * vm.utilization[time_index].maxcpu))
        
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

def balance_cluster(cluster):
    performed_migrations = []
    for i in range(0, 55):
        migrations = find_migrations(cluster)
        _, old_score = cluster.get_cluster_score(0.0)
        current_migration = None
        allocations = []
        migrationsList = [(k, v) for k, v in migrations.items()]
        migrationsList.sort(key=lambda a: a[1][1], reverse=True)
        for j in range(0, 10):
            key = migrationsList[j][0]
            current_migration = (key, migrations[key])
        
            # Perform the current migration for each potential target node
            scores = [(100, 100) for x in range(0, len(cluster.nodes))]
            clusters = []
            for j in range(0, len(cluster.nodes)):
                new_cluster = None
                if j != current_migration[1][0]:
                    new_cluster = deepcopy(cluster)
                    migrate(current_migration[0], new_cluster.nodes[current_migration[1][0]], new_cluster.nodes[j], new_cluster)
                    scores[j] = new_cluster.get_cluster_score(0.0)
                clusters.append(new_cluster)
            # Select the best migration
            max_score = old_score
            best_cluster = cluster
            temp_perf_migrations = None
            for index, (_, score) in enumerate(scores):
                if score < max_score:
                    max_score = score
                    best_cluster = clusters[index]
                    temp_perf_migrations = deepcopy(performed_migrations)
                    temp_perf_migrations.append({"vm_id":current_migration[0], "source_node": clusters[index].nodes[current_migration[1][0]].name, "target_node": clusters[index].nodes[index].name})
            allocations.append((max_score, best_cluster, temp_perf_migrations))
        
        # Select the best of all migrations performed
        max_score = old_score
        for j in range(0, len(allocations)):
            temp_score, temp_cluster, temp_perf_migrations = allocations[j]
            if (temp_score < max_score):
                best_cluster = temp_cluster
                max_score = temp_score
                performed_migrations = temp_perf_migrations
        cluster = deepcopy(best_cluster)
        # del(clusters)
        # print(scores)
        print("Iteration: {0} Score increase: {1}%".format(i, round((max_score-old_score)/old_score * 100, 2)))
        print(performed_migrations)
    return cluster