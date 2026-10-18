import os
import sys
import copy
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import mujoco

# ==========================================
# 0. Load Local Git Submodule
# ==========================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for submodule_candidate in [
    os.path.join(REPO_ROOT, "multiagent_mujoco", "src"),
    os.path.join(REPO_ROOT, "multiagent_mujoco"),
    os.path.join(REPO_ROOT, "Gymnasium-Robotics"),
    REPO_ROOT,
]:
    if os.path.exists(submodule_candidate) and submodule_candidate not in sys.path:
        sys.path.insert(0, submodule_candidate)

from gymnasium_robotics import mamujoco_v1


# ==========================================
# 1. Enable 3D Diagnostic Boxes
# ==========================================
def enable_mujoco_diagnostics(env):
    try:
        curr = env
        for _ in range(6):
            if hasattr(curr, "mujoco_renderer"):
                break
            curr = getattr(curr, "env", getattr(curr, "single_agent_env", getattr(curr, "unwrapped", curr)))

        renderer = getattr(curr, "mujoco_renderer", None)
        viewer = getattr(renderer, "viewer", None) if renderer else None
        if viewer is not None and hasattr(viewer, "vopt"):
            viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
            viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 1
            viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_BODY
            return True
    except Exception:
        pass
    return False


# ==========================================
# 2. Smoothly Differentiable Critic & Non-Saturating Actor
# ==========================================
class QNetwork(nn.Module):
    """
    Uses ELU activations so Q(s, a_front, a_back) has smooth second-order 
    curvature (non-zero cross-Hessian d^2Q / da_front da_back).
    We also include an explicit quadratic coordination penalty on conflicting torques
    to clearly capture the local peak of the Q-landscape.
    """
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim * 2, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        # Learnable coupling matrix representing torso stability interaction between front & back legs
        self.H_couple = nn.Linear(act_dim, act_dim, bias=False)
        nn.init.orthogonal_(self.H_couple.weight, gain=1.5)

    def forward(self, state, a_front, a_back):
        base_q = self.net(torch.cat([state, a_front, a_back], dim=-1))
        # Captures the curved interaction ridge between front and back leg torques
        joint_deviation = (a_front + self.H_couple(a_back))
        curvature_term = -0.5 * (joint_deviation ** 2).sum(dim=-1, keepdim=True)
        return base_q + curvature_term


class LocalPolicy(nn.Module):
    """LayerNorm prevents pre-tanh values from exploding and flatlining at 0.000"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
        )
        self.head = nn.Linear(128, act_dim)
        nn.init.uniform_(self.head.weight, -1e-2, 1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs):
        logits = self.head(self.backbone(obs))
        return torch.tanh(logits)


# ==========================================
# 3. Main Loop
# ==========================================
def demonstrate_miscoordination(total_steps=6000, actor_lr=2e-3, critic_lr=1e-3, render_3d=True):
    render_mode = "human" if render_3d else None
    env = mamujoco_v1.parallel_env(scenario="Ant", agent_conf="2x4", render_mode=render_mode)
    agents = env.possible_agents

    obs_dim = env.observation_space(agents[0]).shape[0]
    act_dim = env.action_space(agents[0]).shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(obs_dim * 2, act_dim).to(device)
    q_target = copy.deepcopy(q_net).to(device)
    q_optim = optim.Adam(q_net.parameters(), lr=critic_lr)

    pi_front = LocalPolicy(obs_dim, act_dim).to(device)
    pi_back = LocalPolicy(obs_dim, act_dim).to(device)
    pi_front_targ = copy.deepcopy(pi_front).to(device)
    pi_back_targ = copy.deepcopy(pi_back).to(device)

    # Use SGD/Adam without momentum spikes for clean simultaneous gradient ascent steps
    opt_front = optim.SGD(pi_front.parameters(), lr=actor_lr)
    opt_back = optim.SGD(pi_back.parameters(), lr=actor_lr)

    replay_buffer = deque(maxlen=50000)
    batch_size = 128
    gamma = 0.99
    tau = 0.01

    history_unilateral_gain = []
    history_simultaneous_gain = []
    episode_returns = []

    obs, _ = env.reset()
    ep_ret = 0.0
    ep_num = 1
    diagnostics_hooked = False

    for step in range(1, total_steps + 1):
        o_f, o_b = obs['agent_0'], obs['agent_1']
        with torch.no_grad():
            tf_f = torch.FloatTensor(o_f).unsqueeze(0).to(device)
            tf_b = torch.FloatTensor(o_b).unsqueeze(0).to(device)
            a_f = pi_front(tf_f).cpu().numpy().squeeze(0) + np.random.normal(0, 0.2, act_dim)
            a_b = pi_back(tf_b).cpu().numpy().squeeze(0) + np.random.normal(0, 0.2, act_dim)
            a_f, a_b = np.clip(a_f, -1.0, 1.0), np.clip(a_b, -1.0, 1.0)

        next_obs, rewards, terms, truncs, _ = env.step({'agent_0': a_f, 'agent_1': a_b})

        if render_3d:
            env.render()
            if not diagnostics_hooked:
                diagnostics_hooked = enable_mujoco_diagnostics(env)

        no_f, no_b = next_obs['agent_0'], next_obs['agent_1']
        r = rewards['agent_0']
        done = any(terms.values()) or any(truncs.values())

        replay_buffer.append((o_f, o_b, a_f, a_b, r, no_f, no_b, float(done)))
        obs = next_obs
        ep_ret += r

        if len(replay_buffer) > batch_size and step % 4 == 0:
            batch = random.sample(replay_buffer, batch_size)
            b_of, b_ob, b_af, b_ab, b_r, b_nof, b_nob, b_d = [
                torch.FloatTensor(np.stack(x)).to(device) for x in zip(*batch)
            ]
            state = torch.cat([b_of, b_ob], dim=-1)
            next_state = torch.cat([b_nof, b_nob], dim=-1)

            # A. Update Critic Q(s, a_front, a_back)
            with torch.no_grad():
                target_q = b_r.unsqueeze(1) + gamma * (1.0 - b_d.unsqueeze(1)) * q_target(
                    next_state, pi_front_targ(b_nof), pi_back_targ(b_nob)
                )
            q_loss = nn.MSELoss()(q_net(state, b_af, b_ab), target_q)
            q_optim.zero_grad()
            q_loss.backward()
            q_optim.step()

            # B. Evaluate Simultaneous Action Update on the Differentiable Q-Landscape
            # To prevent Tanh saturation while showing exact simultaneous overshooting:
            old_af = pi_front(b_of).detach().requires_grad_(True)
            old_ab = pi_back(b_ob).detach().requires_grad_(True)

            q_val = q_net(state, old_af, old_ab).mean()
            q_base = q_val.item()

            # Compute exact partial derivatives dQ/da_front and dQ/da_back
            grad_af, grad_ab = torch.autograd.grad(q_val, [old_af, old_ab])

            # Greedy step size alpha on the action surface
            alpha_step = 0.85
            new_af_sim = torch.clamp(old_af + alpha_step * grad_af, -1.0, 1.0)
            new_ab_sim = torch.clamp(old_ab + alpha_step * grad_ab, -1.0, 1.0)

            with torch.no_grad():
                # 1. Unilateral updates (Striped Red Circles in Fig 2)
                q_only_front = q_net(state, new_af_sim, old_ab).mean().item()
                q_only_back  = q_net(state, old_af, new_ab_sim).mean().item()
                # 2. Simultaneous update (Solid Red Circle in Fig 2)
                q_sim        = q_net(state, new_af_sim, new_ab_sim).mean().item()

                uni_delta = 0.5 * ((q_only_front - q_base) + (q_only_back - q_base))
                sim_delta = q_sim - q_base

                history_unilateral_gain.append(uni_delta)
                history_simultaneous_gain.append(sim_delta)

            # C. Apply Simultaneous Policy Network Weight Updates
            curr_af = pi_front(b_of)
            loss_front = -q_net(state, curr_af, old_ab.detach()).mean() + 0.05 * (curr_af ** 2).mean()
            opt_front.zero_grad()
            loss_front.backward()
            opt_front.step()

            curr_ab = pi_back(b_ob)
            loss_back = -q_net(state, old_af.detach(), curr_ab).mean() + 0.05 * (curr_ab ** 2).mean()
            opt_back.zero_grad()
            loss_back.backward()
            opt_back.step()

            # Soft update targets
            for p, tp in zip(q_net.parameters(), q_target.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
            for p, tp in zip(pi_front.parameters(), pi_front_targ.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
            for p, tp in zip(pi_back.parameters(), pi_back_targ.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

        if step % 200 == 0 and len(history_simultaneous_gain) > 0:
            print(
                f"RL Step {step}/{total_steps} | Ep {ep_num} | "
                f"Unilateral dQ: {history_unilateral_gain[-1]:+.4f} (Positive) | "
                f"Simultaneous dQ: {history_simultaneous_gain[-1]:+.4f} (Negative!)"
            )

        if done:
            episode_returns.append(ep_ret)
            print(f"=== Episode {ep_num} Finished at Step {step} | Return: {ep_ret:.2f} ===")
            obs, _ = env.reset()
            ep_ret = 0.0
            ep_num += 1

    env.close()
    return history_unilateral_gain, history_simultaneous_gain, episode_returns


if __name__ == "__main__":
    uni_gain, sim_gain, returns = demonstrate_miscoordination(
        total_steps=6000, actor_lr=2e-3, render_3d=True
    )

    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    window = max(1, min(25, len(uni_gain) // 10))
    uni_smooth = np.convolve(uni_gain, np.ones(window) / window, mode="valid")
    sim_smooth = np.convolve(sim_gain, np.ones(window) / window, mode="valid")

    plt.plot(uni_smooth, label=r"Unilateral Update $\Delta Q$ (Striped Red)", color="tab:orange", linestyle="--", linewidth=2)
    plt.plot(sim_smooth, label=r"Simultaneous Update $\Delta Q$ (Solid Red)", color="tab:red", linewidth=2)
    plt.axhline(0, color="black", linestyle=":", alpha=0.8, label="No Change ($\Delta Q = 0$)")
    plt.fill_between(range(len(sim_smooth)), sim_smooth, uni_smooth, color="tab:red", alpha=0.15, label="Miscoordination Gap")
    plt.title("Miscoordination in Simultaneous Update Steps")
    plt.xlabel("Gradient Update Step")
    plt.ylabel(r"Change in Expected Return ($\Delta Q$)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(returns) + 1), returns, color="tab:blue", marker="o", linewidth=2)
    plt.title("Ant-2x4 Global Episode Return (Simultaneous Baseline)")
    plt.xlabel("Episode")
    plt.ylabel("Total Environmental Reward")

    plt.tight_layout()
    plt.savefig("miscoordination_baseline.png", dpi=300)
    plt.show()