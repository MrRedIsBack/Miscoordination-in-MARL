import numpy as np

class EmbeddedNodeAgent:
    """
    Represents an embedded IoT sensor node learning to select a TDMA time slot
    to transmit packets without colliding with peer nodes.
    """
    def __init__(self, node_id: int, num_slots: int, lr: float = 0.1):
        self.node_id = node_id
        self.num_slots = num_slots
        self.lr = lr
        # Stateless Q-table: estimates expected reward for each time slot
        self.q_table = np.zeros(num_slots, dtype=np.float64)

    def get_boltzmann_probabilities(self, temperature: float) -> np.ndarray:
        """
        Computes action probabilities using the Boltzmann distribution.
        Includes the max-Q subtraction trick for numerical stability.
        """
        if temperature <= 1e-5:
            # Hard greedy limit when temperature approaches 0
            probs = np.zeros(self.num_slots)
            probs[np.argmax(self.q_table)] = 1.0
            return probs

        # Subtract max(Q) to prevent np.exp() overflow at low temperatures
        scaled_q = (self.q_table - np.max(self.q_table)) / temperature
        exp_q = np.exp(scaled_q)
        return exp_q / np.sum(exp_q)

    def select_slot(self, temperature: float) -> int:
        """Samples a transmission time slot based on Boltzmann probabilities."""
        probs = self.get_boltzmann_probabilities(temperature)
        return int(np.random.choice(self.num_slots, p=probs))

    def update_q(self, slot: int, reward: float):
        """Standard stateless Q-learning update (bandit formulation)."""
        self.q_table[slot] += self.lr * (reward - self.q_table[slot])


class PacketChannelEnv:
    """
    Simulates a shared wireless communication channel over a 1-frame period
    divided into `num_slots` discrete time slots.
    """
    def __init__(self, num_nodes: int, num_slots: int):
        self.num_nodes = num_nodes
        self.num_slots = num_slots

    def step(self, chosen_slots: list[int]) -> tuple[list[float], int]:
        """
        Evaluates packet transmissions across the frame.
        - Unique slot selection -> ACK received (+1.0 reward)
        - Multiple nodes in same slot -> Packet collision (-1.0 reward)
        """
        slot_counts = np.bincount(chosen_slots, minlength=self.num_slots)
        rewards = []
        successful_packets = 0

        for slot in chosen_slots:
            if slot_counts[slot] == 1:
                rewards.append(1.0)   # Successful packet delivery (ACK)
                successful_packets += 1
            else:
                rewards.append(-1.0)  # Collision / NACK (energy wasted)

        return rewards, successful_packets


def run_marl_simulation(
    num_nodes: int = 4,
    num_slots: int = 4,
    episodes: int = 1000,
    temp_start: float = 2.0,
    temp_end: float = 0.05,
    decay_rate: float = 0.995
):
    env = PacketChannelEnv(num_nodes, num_slots)
    agents = [EmbeddedNodeAgent(i, num_slots) for i in range(num_nodes)]
    
    temperature = temp_start
    throughput_history = []

    for ep in range(episodes):
        # 1. Each embedded node independently selects a slot using Boltzmann
        chosen_slots = [agent.select_slot(temperature) for agent in agents]

        # 2. Transmit packets over the shared medium and observe ACK/Collisions
        rewards, successes = env.step(chosen_slots)

        # 3. Update local Q-tables
        for agent, slot, reward in zip(agents, chosen_slots, rewards):
            agent.update_q(slot, reward)

        # 4. Anneal temperature
        temperature = max(temp_end, temperature * decay_rate)
        
        # Track channel utilization efficiency (max possible is num_nodes / num_slots)
        throughput_history.append(successes / num_slots)

    return agents, throughput_history, temperature


# =====================================================================
# Test & Verification Suite
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("--- 1. Numerical Verification of Boltzmann Formula ---")
    test_agent = EmbeddedNodeAgent(node_id=0, num_slots=4)
    test_agent.q_table = np.array([1.0, 0.2, -0.5, -1.0])
    
    for tau in [5.0, 1.0, 0.1]:
        probs = test_agent.get_boltzmann_probabilities(temperature=tau)
        print(f"Temp tau={tau:<4} -> Action Probs: {np.round(probs, 3)} (Sum: {np.sum(probs):.2f})")

    print("\n--- 2. MARL Embedded Packet Scheduling Simulation ---")
    NUM_NODES = 4
    NUM_SLOTS = 4
    agents, history, final_temp = run_marl_simulation(
        num_nodes=NUM_NODES, num_slots=NUM_SLOTS, episodes=1000
    )

    early_tput = np.mean(history[:50]) * 100
    late_tput = np.mean(history[-50:]) * 100
    print(f"Channel Throughput (First 50 frames): {early_tput:.1f}% (High exploration / collisions)")
    print(f"Channel Throughput (Last 50 frames) : {late_tput:.1f}% (Converged orthogonal TDMA)")
    print(f"Final Temperature                   : {final_temp:.4f}\n")

    print("Final Node Allocations & Q-Tables:")
    for agent in agents:
        best_slot = np.argmax(agent.q_table)
        probs = agent.get_boltzmann_probabilities(final_temp)
        print(
            f"  Node {agent.node_id} -> Locked Slot: {best_slot} "
            f"(Prob: {probs[best_slot]*100:5.1f}%) | Q-Values: {np.round(agent.q_table, 2)}"
        )

    # Assert test conditions to verify MARL convergence
    chosen_final_slots = [np.argmax(a.q_table) for a in agents]
    assert len(set(chosen_final_slots)) == NUM_NODES, "Test Failed: Nodes did not converge to unique slots!"
    print("\n[PASS] All embedded nodes successfully converged to collision-free time slots.")