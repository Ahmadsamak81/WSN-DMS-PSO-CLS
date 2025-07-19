import numpy as np
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from DMS_PSO_CLS_LEACH import (
    WirelessSensorNetwork as WSN_DMS, 
    TrustEngine, 
    FitnessFunction as FF_DMS, 
    DMS_PSO_CLS, 
    ClusterManager as CM_DMS, 
    MaliciousNodeManager, 
    TDMACommunicator as TC_DMS, 
    SimulationEngine as SE_DMS
)

from LEACH_PSO import (
    WirelessSensorNetwork as WSN_PSO, 
    FitnessFunction as FF_PSO, 
    PSO_CLS, 
    ClusterManager as CM_PSO, 
    TDMACommunicator as TC_PSO, 
    SimulationEngine as SE_PSO
)

def run_simulation(protocol_name, num_nodes=100, area_size=100, malicious_ratio=0.1, max_rounds=100):
    if protocol_name == "DMS-PSO-CLS-LEACH":
        wsn = WSN_DMS(num_nodes=num_nodes, area_size=area_size, malicious_ratio=malicious_ratio)
        trust_engine = TrustEngine()
        fitness_fn = FF_DMS()  
        optimizer = DMS_PSO_CLS(num_particles=30, num_subswarms=3)
        cluster_manager = CM_DMS()
        malicious_handler = MaliciousNodeManager(malicious_ratio=malicious_ratio)
        communicator = TC_DMS(wsn, cluster_manager, trust_engine)
        simulator = SE_DMS(wsn, optimizer, cluster_manager, malicious_handler, communicator,fitness_fn, max_rounds=max_rounds)
    elif protocol_name == "LEACH-PSO":
        wsn = WSN_PSO(num_nodes=num_nodes, area_size=area_size, malicious_ratio=malicious_ratio)
        fitness_fn = FF_PSO()  
        optimizer = PSO_CLS(num_particles=30)
        cluster_manager = CM_PSO()
        communicator = TC_PSO(wsn, cluster_manager)
        simulator = SE_PSO(wsn, optimizer, cluster_manager, communicator,fitness_fn, max_rounds=max_rounds)
    
    simulator.run()
    return simulator.metrics

def plot_comparison(dms_metrics, pso_metrics):
    rounds_dms = range(1, len(dms_metrics['alive_nodes']) + 1)
    rounds_pso = range(1, len(pso_metrics['alive_nodes']) + 1)

    node_ids = np.arange(len(dms_metrics['ch_selection_counts'][-1]))
    width = 0.35  
    dms_probs = dms_metrics['selection_probability'][-1]  
    pso_probs = pso_metrics['selection_probability'][-1] 
    node_ids = np.arange(len(dms_probs))
    
    plt.figure(figsize=(15, 5))
    
    # Alive Nodes Comparison
    plt.subplot(2, 4, 1)
    plt.plot(rounds_dms, dms_metrics['alive_nodes'], 'b-', label='DMS-PSO-CLS-LEACH')
    plt.plot(rounds_pso, pso_metrics['alive_nodes'], 'r-', label='LEACH-PSO')
    plt.title("Alive Nodes Over Time")
    plt.xlabel("Round")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    
    # Average Energy Comparison
    plt.subplot(2, 4, 2)
    plt.plot(rounds_dms, dms_metrics['energy'], 'b-', label='DMS-PSO-CLS-LEACH')
    plt.plot(rounds_pso, pso_metrics['energy'], 'r-', label='LEACH-PSO')
    plt.title("Average Energy Over Time")
    plt.xlabel("Round")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.grid(True)
    
    # Packets Delivered Comparison
    plt.subplot(2, 4, 3)
    plt.plot(rounds_dms, dms_metrics['packets'], 'b-', label='DMS-PSO-CLS-LEACH')
    plt.plot(rounds_pso, pso_metrics['packets'], 'r-', label='LEACH-PSO')
    plt.title("Packets Delivered Per Round")
    plt.xlabel("Round")
    plt.ylabel("Packets")
    plt.legend()
    plt.grid(True)
    
    # Cumulative Packets Comparison
    plt.subplot(2, 4, 4)
    plt.plot(rounds_dms, dms_metrics['cumulative_packets'], 'b-', label='DMS-PSO-CLS-LEACH')
    plt.plot(rounds_pso, pso_metrics['cumulative_packets'], 'r-', label='LEACH-PSO')
    plt.title("Total Packets Delivered")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Packets")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 4, 5)
    plt.bar(node_ids, dms_probs, color='blue', alpha=0.7)
    plt.title("CH Selection Probability per Node")
    plt.xlabel("Node ID")
    plt.ylabel("Probability")
    plt.grid(True)    

    plt.subplot(2, 4, 6)
    plt.bar(node_ids, pso_probs, color='red', alpha=0.7)
    plt.title("CH Selection Probability per Node")
    plt.xlabel("Node ID")
    plt.ylabel("Probability")
    plt.grid(True)        
    
    # Total Residual Energy Comparison   
    plt.subplot(2, 4, 7)
    plt.plot(rounds_dms, dms_metrics['total_residual_energy'], 'b-', label='DMS-PSO-CLS-LEACH')
    plt.plot(rounds_pso, pso_metrics['total_residual_energy'], 'r-', label='LEACH-PSO')
    plt.title("Total Residual Energy Over Time")
    plt.xlabel("Round")
    plt.ylabel("Total Energy (J)")
    plt.legend()
    plt.grid(True)

    # CH Selection Distribution Comparison
    # plt.subplot(2, 4, 5)
    # if 'ch_selection_counts' in dms_metrics and 'ch_selection_counts' in pso_metrics:
       
    #     # plt.bar(['DMS-PSO-CLS-LEACH', 'LEACH-PSO'], [avg_dms, avg_pso],
    #     #        yerr=[std_dms, std_pso], capsize=10, color=['blue', 'red'])
    #     plt.bar(node_ids - width/2, dms_metrics['ch_selection_counts'][-1], width, label='DMS-PSO-CLS-LEACH', color='blue')
    #     plt.xlabel("Node ID")
    #     plt.ylabel("Times Selected as CH")
    #     plt.title("CH Selection Frequency per Node")
    #     plt.legend()
    #     plt.grid(True)

    # CH Selection Distribution Comparison
    # plt.subplot(2, 4, 6)
    # if 'ch_selection_counts' in dms_metrics and 'ch_selection_counts' in pso_metrics:
      
    #     # plt.bar(['DMS-PSO-CLS-LEACH', 'LEACH-PSO'], [avg_dms, avg_pso],
    #     #        yerr=[std_dms, std_pso], capsize=10, color=['blue', 'red'])
    #     plt.bar(node_ids + width/2, pso_metrics['ch_selection_counts'][-1], width, label='LEACH-PSO', color='red')
    #     plt.xlabel("Node ID")
    #     plt.ylabel("Times Selected as CH")
    #     plt.title("CH Selection Frequency per Node")
    #     plt.legend()
    #     plt.grid(True)    

    # Average CH Energy Comparison
    plt.subplot(2, 4, 8)
    plt.plot(rounds_dms, dms_metrics['avg_ch_energy'], 'b-', label='DMS-PSO-CLS-LEACH')
    plt.plot(rounds_pso, pso_metrics['avg_ch_energy'], 'r-', label='LEACH-PSO')
    plt.title("Average CH Energy Per Round")
    plt.xlabel("Round")
    plt.ylabel("CH Energy (J)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Malicious CHs Comparison (if available)
    # if 'malicious_ch' in pso_metrics and 'malicious_ch_count' in dms_metrics:
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(rounds_dms, [dms_metrics['malicious_ch_count'] / dms_metrics['total_ch_selections'] * 100 if i == len(rounds_dms) - 1 else 0 for i in range(len(rounds_dms))], 
    #              'b-', label='DMS-PSO-CLS-LEACH (Overall %)')
    #     plt.plot(rounds_pso, pso_metrics['malicious_ch'], 'r-', label='LEACH-PSO (Per Round)')
    #     plt.title("Malicious CHs Selected")
    #     plt.xlabel("Round")
    #     plt.ylabel("Count / Percentage")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

if __name__ == "__main__":
    print("#####################################################################################")
    print("num_nodes=100, area_size=100x100, malicious_ratio=10%, packet_size=6000, max_rounds=100")
    print("#####################################################################################")
    print("Running DMS-PSO-CLS-LEACH simulation...")
    dms_metrics = run_simulation("DMS-PSO-CLS-LEACH")
    
    print("\nRunning LEACH-PSO simulation...")
    pso_metrics = run_simulation("LEACH-PSO")
    
    plot_comparison(dms_metrics, pso_metrics)