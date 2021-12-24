import gym
import numpy as np
import ray
from ray import tune
import maddpg
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import argparse
from ray.tune import CLIReporter
import os


class SingleAgentTestEnv(MultiAgentEnv):
    def __init__(
        self,
        max_len=1,
        obs_shape=(1,),
        act_shape = (1,)
    ):
        self.num_agents = 1
        self.agent_id = "agent"
        self.agents = ["agent"]
        self.observation_spaces = {
            "agent": gym.spaces.Box(0, 1, obs_shape)
        }
        self.action_spaces = {"agent": gym.spaces.Box(0, 1, act_shape)}
        self.obs_shape = obs_shape
        self.max_len = max_len

    def reset(self):
        self.dt = 0
        self.obs = np.zeros(self.obs_shape, dtype=np.float32)
        return {self.agent_id: self.obs}

    def step(self, actions):
        self.dt += 1
        action = actions[self.agent_id]
        assert self.dt <= self.max_len, "Environment should be reset on done"
        assert self.action_spaces[self.agent_id].contains(action)
        self.obs = np.zeros(self.obs_shape, dtype=np.float32)
        reward = np.array([1 - action])
        done = np.array([self.dt == self.max_len], dtype=bool)
        return (
            {self.agent_id: self.obs},
            {self.agent_id: reward},
            {self.agent_id: done, "__all__": self.dt == self.max_len},
            {self.agent_id: None},
        )

    def render(self):
        return


class MultiAgentTestEnv(MultiAgentEnv):
    def __init__(self, num_agents=5):
        self.num_agents = num_agents
        self.agents = {"agent_%d" % i for i in range(self.num_agents)}
        self.observation_spaces = {
            agent: gym.spaces.Box(0, 1, (1,)) for agent in self.agents
        }
        self.action_spaces = {agent: gym.spaces.Discrete(2) for agent in self.agents}
        self.obs = np.zeros([self.num_agents])

    def reset(self):
        self.obs = np.zeros([self.num_agents])
        return self.observe()

    def observe(self):
        return {agent: np.array([self.obs[i]]) for i, agent in enumerate(self.agents)}

    def state(self):
        return np.array(self.obs)

    def step(self, actions):
        self.obs = [actions[agent] for agent in self.agents]
        reward = 1 if np.all(self.obs) == 1 else 0
        return (
            self.observe(),
            {agent: reward for agent in self.agents},
            {agent: False for agent in self.agents},
            None,
        )

    def render(self):
        return

    def close(self):
        return


def parse_args():
    # Environment
    parser = argparse.ArgumentParser(
        "RLLib MADDPG with test environments from"
        + "Andy Jones' Debugging Deep RL Advice"
    )

    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "tfe", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="ERROR",
        help="The log level for tune.run()",
    )
    parser.add_argument(
        "--max-episode-len", type=int, default=25, help="maximum episode length"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=60000, help="number of episodes"
    )
    parser.add_argument(
        "--num-adversaries", type=int, default=0, help="number of adversarial agents"
    )
    parser.add_argument(
        "--good-policy", type=str, default="maddpg", help="policy for good agents"
    )
    parser.add_argument(
        "--adv-policy", type=str, default="maddpg", help="policy of adversaries"
    )

    # Core training parameters
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="learning rate for Adam optimizer"
    )
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument(
        "--rollout-fragment-length",
        type=int,
        default=25,
        help="number of data points sampled /update /worker",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1024,
        help="number of data points /update",
    )
    parser.add_argument(
        "--n-step", type=int, default=1, help="length of multistep value backup"
    )
    parser.add_argument(
        "--num-units", type=int, default=64, help="number of units in the mlp"
    )
    parser.add_argument(
        "--replay-buffer",
        type=int,
        default=1000000,
        help="size of replay buffer in training",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="~/ray_results",
        help="path to save checkpoints",
    )

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs-per-worker", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=0)

    return parser.parse_args()


def main(args):
    ray.init()
    MADDPGAgent = maddpg.MADDPGTrainer
    env_name = "test"

    def env_creator(config):
        env = SingleAgentTestEnv()
        return env

    register_env(env_name, lambda config: env_creator(config))
    env = env_creator(None)
    agents = env.agents

    def gen_policy(i):
        use_local_critic = [
            args.adv_policy == "ddpg"
            if i < args.num_adversaries
            else args.good_policy == "ddpg"
            for i in range(len(env.agents))
        ]
        return (
            None,
            env.observation_spaces[agents[i]],
            maddpg._make_continuous_space(env.action_spaces[agents[i]]),
            {
                "agent_id": i,
                "use_local_critic": use_local_critic[i],
            },
        )

    policies = {"policy_%d" % i: gen_policy(i) for i in range(len(env.agents))}
    policy_ids = list(policies.keys())

    config = {
        # === Setup ===
        "framework": args.framework,
        "log_level": args.log_level,
        "env": env_name,
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        "num_envs_per_worker": args.num_envs_per_worker,
        "horizon": args.max_episode_len,
        # === Policy Config ===
        # --- Model ---
        "good_policy": args.good_policy,
        "adv_policy": args.adv_policy,
        "actor_hiddens": [args.num_units] * 2,
        "actor_hidden_activation": "relu",
        "critic_hiddens": [args.num_units] * 2,
        "critic_hidden_activation": "relu",
        "n_step": args.n_step,
        "gamma": args.gamma,
        # --- Exploration ---
        "tau": 0.01,
        # --- Replay buffer ---
        "buffer_size": args.replay_buffer,
        # --- Optimization ---
        "actor_lr": args.lr,
        "critic_lr": args.lr,
        "learning_starts": args.train_batch_size * args.max_episode_len,
        "rollout_fragment_length": args.rollout_fragment_length,
        "train_batch_size": args.train_batch_size,
        "batch_mode": "truncate_episodes",
        # === Multi-agent setting ===
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda name, _: policy_ids[agents.index(name)],
            # Workaround because MADDPG requires agent_id: int but actual ids are strings like 'speaker_0'
        },
    }

    tune.run(
        MADDPGAgent,
        name="Torch_MADDPG",
        config=config,
        progress_reporter=CLIReporter(),
        stop={
            "episodes_total": args.num_episodes,
        },
        local_dir=os.path.join(args.local_dir, env_name),
        verbose=2,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
