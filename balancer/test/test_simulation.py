from copy import deepcopy

import numpy as np
from data_interface import load_cluster_from_file
from util import trim_cluster_data, plot_cluster_cpu_usage, pearson_coefficient
from simulation import migrate

def test_simulation():
    simulation_timeframe = [(48,67),(42,65),(43,70)]
    reference_timeframe = [(50,69),(44,67),(22,49)]

    p_coefficients = []
    for n_scenario in range(1,4):
        base_state = load_cluster_from_file('snapshots/scenario{0}/snapshot-before-hour-AVERAGE'.format(n_scenario))

        expected_state = load_cluster_from_file('snapshots/scenario{0}/snapshot-after-hour-AVERAGE'.format(n_scenario))

        simulated_state = deepcopy(base_state)
        
        # Trim cluster utilization entries to period of interest
        simulated_state = trim_cluster_data(simulated_state, simulation_timeframe[n_scenario-1])
        expected_state = trim_cluster_data(expected_state, reference_timeframe[n_scenario-1])

        _ = simulated_state.get_average_cpu_usage()

        migrate(114, simulated_state.get_node_by_name("pve-g2n6"), simulated_state.get_node_by_name("pve-g2n8"), simulated_state)

        _ = simulated_state.get_average_cpu_usage()
        _ = expected_state.get_average_cpu_usage()

        plot_cluster_cpu_usage(simulated_state, ["pve-g2n6", "pve-g2n8"])
        plot_cluster_cpu_usage(expected_state, ["pve-g2n6", "pve-g2n8"])

        utilization_reference_node1 = expected_state.get_node_by_name("pve-g2n6")
        utilization_expected_node1 = simulated_state.get_node_by_name("pve-g2n6")

        utilization_reference_node2 = expected_state.get_node_by_name("pve-g2n8")
        utilization_expected_node2 = simulated_state.get_node_by_name("pve-g2n8")

        r1 = pearson_coefficient(utilization_reference_node1.aggregate_utilization, utilization_expected_node1.aggregate_utilization)
        r2 = pearson_coefficient(utilization_reference_node2.aggregate_utilization, utilization_expected_node2.aggregate_utilization)

        p_coefficients.append(r1)
        p_coefficients.append(r2)
    
    print_testing_results(p_coefficients)

def print_testing_results(p_coefficients):
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
    print("Total avg accuracy: {0}".format(np.average(p_coefficients)))