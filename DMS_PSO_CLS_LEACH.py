
############################################
###### In this code we solve node death Management + Energy Distribution
############################################
import numpy as np
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

class Node:
    def __init__(self, node_id, x, y, initial_energy=0.05, is_malicious=False):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.energy = max(0, initial_energy)  
        self.initial_energy = initial_energy
        self.energy_threshold = 0.01  
        self.is_malicious = is_malicious
        self.trust = 1.0 if not is_malicious else 0.3
        self.trust_history = []
        self.packet_success = 0
        self.packet_fail = 0
        self.packets_forwarded = 0
        self.dropped_packets = 0
        self.is_alive = True
        self.is_CH = False
        self.cluster_members = []
        self.ch_id = None
        self.tdma_slot = None
        self.cooldown = 0
        self.CH_count = 0

    def distance_to(self, other_node):
        return math.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)

    def update_trust(self):
        if self.is_malicious:
            self.trust = max(0, self.trust - 0.1)
        else:
            total = self.packet_success + self.packet_fail
            self.trust = max(0, min(1, self.packet_success / total)) if total > 0 else 1.0
        self.trust_history.append(self.trust)

    def consume_energy(self, distance, is_CH=False, packet_size=6000):
        E_elec = 50e-9
        E_fs = 10e-12
        E_mp = 0.0013e-12
        E_DA = 5e-9
        d0 = math.sqrt(E_fs / E_mp)
        # After the energy dissipated in a given node reached a set threshold,
        # that node was considered dead for the remainder of the simulation.
        energy_threshold = 0.01

        if is_CH:
            # Cluster head energy consumption
            k = len(self.cluster_members)
            if distance < d0:
                tx_energy = k * E_elec + k * E_fs * distance**2
            else:
                tx_energy = k * E_elec + k * E_mp * distance**4
            agg_energy = k * E_DA * packet_size
            total_energy = tx_energy + agg_energy
        else:
            # Member node energy consumption
            if distance < d0:
                total_energy = packet_size * (E_elec + E_fs * distance**2)
            else:
                total_energy = packet_size * (E_elec + E_mp * distance**4)

        self.energy -= total_energy
        if self.energy <= energy_threshold: 
            self.is_alive = False
            self.energy = 0
            self.is_CH = False 

        return total_energy

class WirelessSensorNetwork:
    def __init__(self, num_nodes=100, area_size=100, initial_energy=0.05, malicious_ratio=0.1, bs_position='center'):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.round = 0 
        self.nodes = []
        self.bs = self.set_base_station(bs_position)
        self.malicious_ids = []
        self.init_nodes(initial_energy, malicious_ratio)

    def set_base_station(self, position):
        if position == 'center':
            return Node('BS', self.area_size/2, self.area_size/2, float('inf'))
        elif position == 'edge':
            return Node('BS', self.area_size/2, self.area_size+10, float('inf'))
        else:
            raise ValueError("Base station position must be 'center' or 'edge'")

    def init_nodes(self, initial_energy, malicious_ratio):
        num_malicious = int(self.num_nodes * malicious_ratio)
        malicious_ids = random.sample(range(self.num_nodes), num_malicious)
        self.malicious_ids = malicious_ids

        for i in range(self.num_nodes):
            x = random.uniform(0, self.area_size)
            y = random.uniform(0, self.area_size)
            is_malicious = i in malicious_ids
            node = Node(i, x, y, initial_energy, is_malicious)
            self.nodes.append(node)

    def get_alive_nodes(self):
        return [node for node in self.nodes if node.is_alive]

    def get_trusted_nodes(self, threshold=0.5):
        return [node for node in self.get_alive_nodes() if node.trust >= threshold and not node.is_malicious]

    def reset_round_states(self):
        for node in self.nodes:
            node.is_CH = False  
            node.cluster_members = []
            node.ch_id = None
            node.tdma_slot = None

class TrustEngine:
    def __init__(self, trust_threshold=0.5, window_size=10):
        self.trust_threshold = trust_threshold
        self.window_size = window_size

    def update_trust_scores(self, wsn):
        for node in wsn.get_alive_nodes():
            node.update_trust()
            if len(node.trust_history) > self.window_size:
                node.trust_history.pop(0)

    def detect_malicious_nodes(self, wsn):
        flagged_nodes = []
        for node in wsn.get_alive_nodes():
            if node.trust < self.trust_threshold or node.is_malicious:
                flagged_nodes.append(node.node_id)
        return flagged_nodes

    def record_packet_result(self, sender_node, success):
        if not sender_node.is_alive:
            return
        if success:
            sender_node.packet_success += 1
        else:
            sender_node.packet_fail += 1

class FitnessFunction:
    def __init__(self):
        self.weights = {
            'energy': 0.3,
            'distance': 0.2,
            'density': 0.1,
            'trust': 0.2,
            'pdr': 0.1,
            'avg_ch_trust': 0.1
        }

    def normalize(self, value, max_value):
        return value / max_value if max_value > 0 else 0

    def compute_density(self, node, wsn, radius=15):
        neighbors = sum(
            1 for other in wsn.get_alive_nodes()
            if other.node_id != node.node_id and node.distance_to(other) <= radius
        )
        return neighbors

    def compute_fitness(self, node, wsn, avg_ch_trust, pd_history):
        if not node.is_alive or node.cooldown > 0 or node.energy < node.energy_threshold:
            return -1

        RE = self.normalize(node.energy, node.initial_energy)
        DBS = self.normalize(node.distance_to(wsn.bs), math.sqrt(2) * wsn.area_size)
        ND = self.normalize(self.compute_density(node, wsn), wsn.num_nodes)
        TL = node.trust
        PD = self.normalize(pd_history.get(node.node_id, 1), 1.0)
        ATCH = avg_ch_trust

        fitness = (
            self.weights['energy'] * RE -
            self.weights['distance'] * DBS +
            self.weights['density'] * ND +
            self.weights['trust'] * TL +
            self.weights['pdr'] * PD +
            self.weights['avg_ch_trust'] * ATCH
        )
        return max(0, min(1, fitness))

    def adapt_weights(self, metrics_over_time):
        if len(metrics_over_time) < 3:
            return

        recent = metrics_over_time[-1]
        prev = metrics_over_time[-2]

        if recent['avg_ch_trust'] < prev['avg_ch_trust']:
            self.weights['trust'] += 0.02
            self.weights['avg_ch_trust'] += 0.02

        if recent['pdr'] < prev['pdr']:
            self.weights['pdr'] += 0.02

        if recent['avg_energy'] < prev['avg_energy']:
            self.weights['energy'] += 0.03

        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total

class Particle:
    def __init__(self, position, velocity, fitness=0):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_fitness = fitness

class DMS_PSO_CLS:
    def __init__(self, num_particles=20, max_iters=15, num_clusters=5, inertia=0.5, c1=1.5, c2=1.5, num_subswarms=5):
        self.num_particles = num_particles
        self.max_iters = max_iters
        self.num_clusters = num_clusters
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.num_subswarms = num_subswarms

    def initialize_particles(self, num_nodes, fitness_fn, wsn, avg_ch_trust, pd_history):
        particles = []
        for _ in range(self.num_particles):
            position = np.zeros(num_nodes)
            eligible_nodes = [i for i, node in enumerate(wsn.nodes) 
                           if node.is_alive and node.energy > node.energy_threshold and node.cooldown == 0 and node.trust >= 0.4]
            
            if not eligible_nodes:
                eligible_nodes = [i for i, node in enumerate(wsn.nodes) if node.is_alive and node.energy > node.energy_threshold]
            
            ch_indices = np.random.choice(eligible_nodes, min(self.num_clusters, len(eligible_nodes)), replace=False).astype(int)
            position[ch_indices] = 1
            velocity = np.random.rand(num_nodes)
            fitness = self.evaluate(position, fitness_fn, wsn, avg_ch_trust, pd_history)
            particles.append(Particle(position, velocity, fitness))
        return particles

    def evaluate(self, position, fitness_fn, wsn, avg_ch_trust, pd_history):
        score = 0
        for i, bit in enumerate(position):
            if bit == 1 and wsn.nodes[i].is_alive and wsn.nodes[i].cooldown == 0:
                score += fitness_fn.compute_fitness(wsn.nodes[i], wsn, avg_ch_trust, pd_history)
        return score

    def update_velocity(self, particle, global_best, inertia, c1, c2):
        r1 = np.random.rand(len(particle.velocity))
        r2 = np.random.rand(len(particle.velocity))
        particle.velocity = (
            inertia * particle.velocity +
            c1 * r1 * (particle.best_position - particle.position) +
            c2 * r2 * (global_best - particle.position)
        )

    def update_position(self, particle, num_clusters):
        probs = 1 / (1 + np.exp(-particle.velocity))
        eligible_indices = [i for i, val in enumerate(probs) 
                          if particle.position[i] == 0 or particle.position[i] == 1]
        if not eligible_indices:
            eligible_indices = range(len(probs))
            
        top_k = min(num_clusters, len(eligible_indices))
        indices = np.argpartition(probs[eligible_indices], -top_k)[-top_k:]
        new_position = np.zeros_like(particle.position)
        new_position[np.array(eligible_indices)[indices]] = 1
        particle.position = new_position

    def run(self, wsn, fitness_fn, avg_ch_trust, pd_history):
        num_nodes = len(wsn.nodes)
        particles = self.initialize_particles(num_nodes, fitness_fn, wsn, avg_ch_trust, pd_history)
        subswarms = [particles[i::self.num_subswarms] for i in range(self.num_subswarms)]

        global_best = None
        global_best_score = -np.inf

        for iteration in range(self.max_iters):
            w = self.inertia * (1 - iteration/self.max_iters)
            # w = 0.9 - iteration * ((0.9 - 0.4) / self.max_iters)
            for swarm in subswarms:
                for particle in swarm:
                    fitness = self.evaluate(particle.position, fitness_fn, wsn, avg_ch_trust, pd_history)
                    if fitness > particle.best_fitness:
                        particle.best_fitness = fitness
                        particle.best_position = particle.position.copy()

                best_in_swarm = max(swarm, key=lambda p: p.best_fitness)
                if best_in_swarm.best_fitness > global_best_score:
                    global_best_score = best_in_swarm.best_fitness
                    global_best = best_in_swarm.best_position.copy()

            for swarm in subswarms:
                for particle in swarm:
                    self.update_velocity(particle, global_best, w, self.c1, self.c2)
                    self.update_position(particle, self.num_clusters)

            if iteration % 5 == 0:
                random.shuffle(particles)
                subswarms = [particles[i::self.num_subswarms] for i in range(self.num_subswarms)]

        return np.where(global_best == 1)[0].astype(int)

class ClusterManager:
    def __init__(self):
        self.clusters = defaultdict(list)
        self.schedule = {}
        self.current_round_ch_ids = []  

    def form_clusters(self, ch_indices, wsn):
        self.clusters.clear()
        self.schedule.clear()
        self.current_round_ch_ids = ch_indices.copy()  

        
        ch_nodes = [wsn.nodes[i] for i in ch_indices if wsn.nodes[i].is_alive and wsn.nodes[i].energy > wsn.nodes[i].energy_threshold and wsn.nodes[i].cooldown == 0 and not wsn.nodes[i].is_malicious]
  
        for i, node in enumerate(wsn.nodes):
            if not node.is_alive:
                continue

            if i in ch_indices:
                node.is_CH = True
                node.CH_count += 1
                node.cooldown = 3
                self.clusters[i].append(i)
                continue

            nearest_ch = min(
                ch_nodes,
                key=lambda ch: node.distance_to(ch),
                default=None
            )
            if nearest_ch:
                self.clusters[nearest_ch.node_id].append(i)
                node.ch_id = nearest_ch.node_id

        # if  wsn.round % 10 == 0 or wsn.round == 1: 
        #     print(f"\nCluster Heads for this round {wsn.round}: {ch_indices}")
        #     for ch_id in ch_indices:
        #         ch = wsn.nodes[ch_id]
        #         print(f"CH {ch_id}: Energy={ch.energy:.4f}, Members={len(self.clusters[ch_id])-1}")
        
        self._assign_tdma_slots()

    def _assign_tdma_slots(self):
        for ch_id, members in self.clusters.items():
            for slot, node_id in enumerate(members):
                self.schedule[node_id] = slot

    def get_tdma_slot(self, node_id):
        return self.schedule.get(node_id, -1)

    def get_cluster_heads(self):
        return list(self.clusters.keys())

class MaliciousNodeManager:
    def __init__(self, malicious_ratio=0.1):
        self.malicious_ratio = malicious_ratio
        self.malicious_nodes = set()

    def inject_malicious_nodes(self, wsn):
        total_nodes = len(wsn.nodes)
        num_malicious = int(total_nodes * self.malicious_ratio)
        candidates = [n.node_id for n in wsn.nodes if n.is_alive and not n.is_malicious]
        selected = random.sample(candidates, min(num_malicious, len(candidates)))
        
        for node_id in selected:
            wsn.nodes[node_id].is_malicious = True
            self.malicious_nodes.add(node_id)

    def detect_malicious_nodes(self, wsn, trust_threshold=0.3, drop_threshold=5):
        detected = []
        actual_malicious = set(wsn.malicious_ids)  
        
        for node in wsn.nodes:
            if not node.is_alive:
                continue
                
            
            if node.node_id in actual_malicious:
                if not node.is_malicious:
                    node.is_malicious = True
                    detected.append(node.node_id)
                continue
                
            
            if (node.trust < trust_threshold and 
                node.dropped_packets > drop_threshold and
                node.packets_forwarded > 10):
                node.is_malicious = True
                detected.append(node.node_id)
        
        return detected

    
class TDMACommunicator:
    def __init__(self, wsn, cluster_manager, trust_engine):
        self.wsn = wsn
        self.cluster_manager = cluster_manager
        self.trust_engine = trust_engine
        self.E_elec = 50e-9
        self.E_fs = 10e-12
        self.E_mp = 0.0013e-12
        self.E_DA = 5e-9
        self.packet_size = 6000
        self.d0 = math.sqrt(self.E_fs / self.E_mp)

    def simulate_round(self):
        total_energy = 0
        total_packets = 0
        total_packets_sent = 0 

        for ch_id, members in self.cluster_manager.clusters.items():
            ch_node = self.wsn.nodes[ch_id]
            if not ch_node.is_alive or ch_node.energy <= ch_node.energy_threshold:  
                print(f"Ignoring dead/low-energy CH: {ch_id}, Energy={ch_node.energy:.4f}")
                ch_node.is_alive = False  
                ch_node.is_CH = False
                continue

            aggregated_data = 0
            for node_id in members:
                node = self.wsn.nodes[node_id]
                if not node.is_alive or node.energy <= 0: 
                    continue

                total_packets_sent += 1  
                distance = node.distance_to(ch_node)
                energy_used = self._transmit_energy(distance)
                node.energy -= energy_used
                total_energy += energy_used

                if random.random() > 0.1:  # 90% delivery rate for honest nodes
                    node.packets_forwarded += 1
                    self.trust_engine.record_packet_result(node, True)
                    aggregated_data += 1
                else:
                    node.dropped_packets += 1
                    self.trust_engine.record_packet_result(node, False)

            if aggregated_data > 0 and ch_node.is_alive and ch_node.energy > 0: 
                distance_to_bs = ch_node.distance_to(self.wsn.bs)
                tx_energy = self._transmit_energy(distance_to_bs)
                agg_energy = aggregated_data * self.E_DA * self.packet_size
                ch_node.energy -= (tx_energy + agg_energy)
                total_energy += (tx_energy + agg_energy)
                total_packets += aggregated_data

            
            for node_id in members + [ch_id]:
                node = self.wsn.nodes[node_id]
                if node.energy <= 0 and node.is_alive:
                    node.is_alive = False
                    node.is_CH = False if node.node_id == ch_id else node.is_CH
                    print(f"Node {node.node_id} died during transmission! Energy={node.energy:.4f}")

        return total_energy, total_packets, total_packets_sent 

    def _transmit_energy(self, distance):
        if distance < self.d0:
            return self.packet_size * (self.E_elec + self.E_fs * distance**2)
        else:
            return self.packet_size * (self.E_elec + self.E_mp * distance**4)

class SimulationEngine:
    def __init__(self, wsn, optimizer, cluster_manager, malicious_handler, communicator,fitness_fn, max_rounds=100,):
        self.wsn = wsn
        self.optimizer = optimizer
        self.fitness_fn = fitness_fn  # 
        self.cluster_manager = cluster_manager
        self.malicious_handler = malicious_handler
        self.communicator = communicator
        self.round = 0
        self.cumulative_packets = 0  
        self.max_rounds = max_rounds
        self.malicious_ch_count = 0
        self.total_ch_selections = 0
        self.ch_selection_count = [0] * self.wsn.num_nodes  
        self.metrics = {
            'alive_nodes': [],
            'energy': [],
            'trust': [],
            'packets': [],
            'cumulative_packets': [],
            'malicious': [],
            'weights': [],
            'malicious_ch': [],
            'total_residual_energy':[],
            'avg_ch_energy':[],
            'ch_selection_counts':[],
            'selection_probability': [],  
            'total_packets_sent': 0,  
            'overall_pdr': []         
        }

    def run(self):
        
        for r in range(1, self.max_rounds+1):
            self.round = r
            self.wsn.round = r

            
            for node in self.wsn.nodes:
                if node.energy <= node.energy_threshold and node.is_alive:
                    node.is_alive = False
                    node.is_CH = False  
            
            # Update cooldowns
            for node in self.wsn.nodes:
                if node.cooldown > 0:
                    node.cooldown -= 1

            # Detect malicious nodes
            self.malicious_handler.detect_malicious_nodes(self.wsn)
            
            # Select CHs using DMS-PSO-CLS
            avg_ch_trust = sum(node.trust for node in self.wsn.nodes if node.is_alive) / len(self.wsn.nodes)
            pd_history = {node.node_id: node.packets_forwarded/(node.packets_forwarded+node.dropped_packets+1e-6)
                         for node in self.wsn.nodes}
            
            ch_indices = self.optimizer.run(self.wsn, self.fitness_fn, avg_ch_trust, pd_history)
            
            
            current_malicious_ch = sum(1 for ch_id in ch_indices if self.wsn.nodes[ch_id].is_malicious)
            self.malicious_ch_count += current_malicious_ch
            self.total_ch_selections += len(ch_indices)

            for node in self.wsn.nodes:
                if node.is_alive and node.is_CH:
                    self.ch_selection_count[node.node_id] += 1

            # Form clusters
            self.cluster_manager.form_clusters(ch_indices, self.wsn)
            
            # Simulate communication
            energy_used, packets_delivered, packets_sent  = self.communicator.simulate_round()
            self.cumulative_packets += packets_delivered
            
            
            total_rounds = r  
            selection_probability = [count / total_rounds for count in self.ch_selection_count]
                       

            # Update metrics
            alive_nodes = sum(1 for node in self.wsn.nodes if node.is_alive)
            avg_energy = sum(node.energy for node in self.wsn.nodes if node.is_alive) / max(1, alive_nodes)
            avg_trust = sum(node.trust for node in self.wsn.nodes if node.is_alive) / max(1, alive_nodes)
            total_residual_energy = sum(node.energy for node in self.wsn.nodes if node.is_alive) 
            avg_ch_energy = np.mean([node.energy for node in self.wsn.nodes if node.is_alive and node.is_CH]) if any(node.is_CH for node in self.wsn.nodes) else 0           
            total_malicious = len(self.malicious_handler.malicious_nodes)

            self.metrics['alive_nodes'].append(alive_nodes)
            self.metrics['energy'].append(avg_energy)
            self.metrics['trust'].append(avg_trust)
            self.metrics['packets'].append(packets_delivered)
            self.metrics['cumulative_packets'].append(self.cumulative_packets)
            self.metrics['malicious'].append(total_malicious)
            self.metrics['weights'].append(self.fitness_fn.weights.copy())
            self.metrics['malicious_ch'].append(current_malicious_ch)
            self.metrics['total_residual_energy'].append(total_residual_energy)
            self.metrics['avg_ch_energy'].append(avg_ch_energy)
            self.metrics['ch_selection_counts'].append(self.ch_selection_count.copy())
            self.metrics['selection_probability'].append(selection_probability)
            self.metrics['overall_pdr'].append(packets_delivered / packets_sent if packets_sent > 0 else 0)
           
            if alive_nodes == 0:
                print("All nodes dead. Simulation ended.")
                break
            # if r % 10 == 0 or r == 1: 
            #     print(f"\nRound {r} - Node Status:")
            #     print("ID | Is_CH | Is_Alive | Energy  | Trust  | Packets | Malicious")
            #     print("-----------------------------------------------------------")
            #     for node in self.wsn.nodes[:10]: 
            #         print(f"{node.node_id:2} | {str(node.is_CH):5} | {str(node.is_alive):7} | {node.energy:.4f} | {node.trust:.4f} | {node.packets_forwarded:7} | {str(node.is_malicious):9}")

        self.print_summary()
        # self.plot_results()

    def print_summary(self):
        malicious_ch_percentage = (self.malicious_ch_count / self.total_ch_selections) * 100 if self.total_ch_selections > 0 else 0

        print("\nDetailed Simulation Summary:")
        print("Round | Alive Nodes | Energy  | Trust  | Packets | Malicious CHs")
        print("-----------------------------------------------------------")
        for r in range(len(self.metrics['alive_nodes'])):
            print(f"{r+1:5} | {self.metrics['alive_nodes'][r]:11} | {self.metrics['energy'][r]:.4f} | "
                  f"{self.metrics['trust'][r]:.4f} | {self.metrics['packets'][r]:7} | "
                  f"{self.metrics['malicious_ch'][r]:9}")
        
        print("\nFinal Statistics:")
        first_death = next((i+1 for i, x in enumerate(self.metrics['alive_nodes']) if x < self.wsn.num_nodes), 'N/A')
        print(f"First node death at round: {first_death}")
        print(f"Last node death at round: {len(self.metrics['alive_nodes']) if self.metrics['alive_nodes'][-1] == 0 else 'N/A'}")
        print(f"Total packets delivered: {sum(self.metrics['packets'])}")
        print(f"Average Packet Delivery Rate (PDR): {np.mean(self.metrics['overall_pdr']):.4f}")
        print(f"Average trust level: {np.mean(self.metrics['trust']):.4f}")
        print(f"Malicious nodes were selected as CHs {self.malicious_ch_count} times ({malicious_ch_percentage:.2f}% of all CH selections)")

    def plot_results(self):
        rounds = range(1, len(self.metrics['alive_nodes'])+1)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2,3,1)
        plt.plot(rounds, self.metrics['alive_nodes'])
        plt.title("Alive Nodes Over Time")
        plt.xlabel("Round")
        plt.ylabel("Count")
        
        plt.subplot(2,3,2)
        plt.plot(rounds, self.metrics['energy'])
        plt.title("Average Energy")
        plt.xlabel("Round")
        plt.ylabel("Energy (J)")
        
        plt.subplot(2,3,3)
        plt.plot(rounds, self.metrics['trust'])
        plt.title("Average Trust")
        plt.xlabel("Round")
        plt.ylabel("Trust Level")
        
        plt.subplot(2,3,4)
        plt.plot(rounds, self.metrics['packets'])
        plt.title("Packets Delivered")
        plt.xlabel("Round")
        plt.ylabel("Packets")
        
        plt.subplot(2,3,5)
        plt.plot(rounds, self.metrics['malicious'])
        plt.title("Malicious Nodes Detected")
        plt.xlabel("Round")
        plt.ylabel("Count")
        
        plt.subplot(2,3,6)
        plt.plot(rounds, self.metrics['cumulative_packets'])
        plt.title("Packets Delivered")
        plt.xlabel("Round")
        plt.ylabel("Packets")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

class Visualizer:
    def __init__(self, wsn):
        self.wsn = wsn
        

    def plot_network(self, cluster_manager=None, current_round_ch_ids=None):
        plt.figure(figsize=(10, 8))
        cmap = plt.cm.get_cmap('RdYlGn')
        self.cluster_manager = cluster_manager  
               
        # Plot normal nodes
        normal_nodes = [n for n in self.wsn.nodes if n.is_alive and not n.is_malicious and not n.is_CH]
        plt.scatter([n.x for n in normal_nodes], [n.y for n in normal_nodes],
                   c=[n.trust for n in normal_nodes], cmap=cmap, vmin=0, vmax=1,
                   s=50, edgecolors='k', label='Normal Nodes')
        
        
        # Plot CHs
        ch_nodes = [n for n in self.wsn.nodes if n.is_alive and n.node_id in self.cluster_manager.current_round_ch_ids]
        plt.scatter([n.x for n in ch_nodes], [n.y for n in ch_nodes],
                   c=[n.trust for n in ch_nodes], cmap=cmap, vmin=0, vmax=1,
                   s=150, marker='^', edgecolors='k', label='Cluster Heads')
        
        # Plot malicious nodes
        malicious_nodes = [n for n in self.wsn.nodes if n.is_alive and n.is_malicious]
        plt.scatter([n.x for n in malicious_nodes], [n.y for n in malicious_nodes],
                   c='black', marker='x', s=100, label='Malicious Nodes')
        
        # Plot dead nodes
        dead_nodes = [n for n in self.wsn.nodes if not n.is_alive]
        plt.scatter([n.x for n in dead_nodes], [n.y for n in dead_nodes],
                   c='gray', marker='s', s=30, label='Dead Nodes')
        
        # Plot base station
        plt.scatter(self.wsn.bs.x, self.wsn.bs.y, c='gold', marker='*', s=300, label='Base Station')
        
        # Draw cluster connections
        if cluster_manager:
            for ch_id, members in cluster_manager.clusters.items():
                ch = self.wsn.nodes[ch_id]
                for member_id in members:
                    member = self.wsn.nodes[member_id]
                    if member.is_alive:
                        plt.plot([ch.x, member.x], [ch.y, member.y], 'b--', alpha=0.3)
        
        plt.colorbar(label='Trust Level')
        plt.title("WSN Topology with Trust Levels")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

# Main execution
if __name__ == "__main__":
    # Initialize components
    wsn = WirelessSensorNetwork(num_nodes=100, area_size=100, malicious_ratio=0.05)
    trust_engine = TrustEngine()
    fitness_fn = FitnessFunction()
    optimizer = DMS_PSO_CLS(num_particles=30, num_subswarms=3)
    cluster_manager = ClusterManager()
    malicious_handler = MaliciousNodeManager()
    communicator = TDMACommunicator(wsn, cluster_manager, trust_engine)
    
    # Run simulation
    simulator = SimulationEngine(wsn, optimizer, cluster_manager, malicious_handler, communicator,fitness_fn, max_rounds=100)
    simulator.run()
    simulator.print_summary()  
    
    # Visualize final network state
    visualizer = Visualizer(wsn)
    visualizer.plot_network(cluster_manager)