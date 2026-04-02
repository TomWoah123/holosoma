import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional, Sequence, Union, TypedDict
import math

from holosoma.agents.fpo.solver import ODESolver
from holosoma.agents.fpo.path import CondOTProbPath
from holosoma.agents.base_algo.base_algo import BaseAlgo

# Additional imports for PPO-like structure
from loguru import logger
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import attach_onnx_metadata, export_policy_as_onnx, get_command_ranges_from_env, get_control_gains_from_config, get_urdf_text_from_robot_config

console = Console()

SMPL_MUJOCO_NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Toe", "R_Toe", "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"]

class FPOConfig(TypedDict):
    hidden_size: int
    parameterization: str
    solver_step_size: float
    perturb_action_std: float
    prior_noise_std: float
    zero_action_input: bool
    condition_drop_ratio: float
    num_sampled_t: int
    num_envs: int
    sample_t_strategy: str
    p_mean: float
    p_std: float
    soft_dropout: bool
    root_track: bool
    hand_track: bool
    actor_learning_rate: float
    critic_learning_rate: float
    num_steps_per_env: int
    num_learning_iterations: int
    num_mini_batches: int
    num_learning_epochs: int
    gamma: float
    lam: float
    value_loss_coef: float
    max_grad_norm: float
    save_interval: int

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class adaLN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm1(x) * (1+scale) + shift
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class FPO(BaseAlgo):
    config: FPOConfig

    def __init__(self, env: BaseTask, config: FPOConfig, log_dir, device="cpu", multi_gpu_cfg: dict | None = None):
        super().__init__(env, config)
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=self.config['num_steps_per_env'],
            num_learning_iterations=self.config['num_learning_iterations'],
            is_main_process=True,  # Assume single GPU for simplicity
            num_gpus=1,
        )

        self._init_config()

        self.current_learning_iteration = 0
        self.eval_callbacks: list[RLEvalCallback] = []
        _ = self.env.reset_all()

    def _init_config(self) -> None:
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()
        self.input_size = self.algo_obs_dim_dict['obs']
        self.action_size = self.env.robot_config.actions_dim

        # Set config attributes
        self.hidden_size = self.config['hidden_size']
        self.parameterization = self.config['parameterization']
        self.solver_step_size = self.config['solver_step_size']
        self.perturb_action_std = torch.tensor(self.config['perturb_action_std'])
        self.prior_noise_std = torch.tensor(self.config['prior_noise_std'])
        self.zero_action_input = self.config['zero_action_input']
        self.condition_drop_ratio = self.config['condition_drop_ratio']
        self.sample_t_strategy = self.config['sample_t_strategy']
        self.p_mean = self.config['p_mean']
        self.p_std = self.config['p_std']
        self.soft_dropout = self.config['soft_dropout']
        self.root_track = self.config['root_track']
        self.hand_track = self.config['hand_track']

        self.num_steps_per_env = self.config['num_steps_per_env']
        self.num_learning_iterations = self.config['num_learning_iterations']
        self.num_mini_batches = self.config['num_mini_batches']
        self.num_learning_epochs = self.config['num_learning_epochs']
        self.gamma = self.config['gamma']
        self.lam = self.config['lam']
        self.value_loss_coef = self.config['value_loss_coef']
        self.max_grad_norm = self.config['max_grad_norm']
        self.save_interval = self.config['save_interval']

    def setup(self):
        logger.info("Setting up FPO")
        self._setup_models_and_optimizer()
        logger.info("Setting up Storage")
        self._setup_storage()

    def _setup_models_and_optimizer(self):
        # Actor MLP
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size + self.action_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, self.hidden_size)),
        )
        self.actor_norm = adaLN(self.hidden_size)
        self.post_adaln_non_linearity = nn.SiLU()
        nn.init.constant_(self.actor_norm.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.actor_norm.adaLN_modulation[-1].bias, 0)
        self.mu = layer_init(nn.Linear(self.hidden_size, self.action_size))

        # Critic MLP
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
            layer_init(nn.Linear(self.hidden_size, 1), std=0.01),
        )

        # Noise embedder
        self.noise_emb = TimestepEmbedder(self.hidden_size)

        # Flow matching stuff
        self.solver = ODESolver()
        self.path = CondOTProbPath()
        self.relative_pose_dropout_mask = self.create_relative_pose_dropout_mask(include_action=True, root_track=self.root_track, hand_track=self.hand_track)
        self.sample_mask = torch.bernoulli(torch.ones(self.config['num_envs'], 1) * self.condition_drop_ratio) * torch.ones(self.config['num_envs'], self.input_size + self.action_size)

        # Optimizers
        self.actor_optimizer = instantiate(
            {"class": "torch.optim.Adam", "lr": self.config['actor_learning_rate']},
            params=list(self.actor_mlp.parameters()) + list(self.actor_norm.parameters()) + list(self.mu.parameters()) + list(self.noise_emb.parameters()),
        )
        self.critic_optimizer = instantiate(
            {"class": "torch.optim.Adam", "lr": self.config['critic_learning_rate']},
            params=self.critic_mlp.parameters(),
        )

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.config['num_steps_per_env'], device=self.device)
        self.storage.register("obs", shape=(self.input_size,), dtype=torch.float)
        self.storage.register("actions", shape=(self.action_size,), dtype=torch.float)
        self.storage.register("rewards", shape=(1,), dtype=torch.float)
        self.storage.register("dones", shape=(1,), dtype=torch.bool)
        self.storage.register("values", shape=(1,), dtype=torch.float)
        self.storage.register("returns", shape=(1,), dtype=torch.float)
        self.storage.register("advantages", shape=(1,), dtype=torch.float)

    def sample_noise(self, noise_shape, device):
        noise = torch.randn(noise_shape, dtype=torch.float32, device=device)
        if self.parameterization == "velocity":
            noise = noise * self.prior_noise_std.to(device)
        return noise

    def create_relative_pose_dropout_mask(self, include_action=True, root_track=False, hand_track=False):
        state_mask = torch.ones(1, 358)
        left_hand_id = SMPL_MUJOCO_NAMES.index("L_Wrist")
        right_hand_id = SMPL_MUJOCO_NAMES.index("R_Wrist")
        assert len(SMPL_MUJOCO_NAMES) == 24

        diff_local_body_pos_flat_mask = torch.zeros(1, 24 * 3)
        diff_local_body_pos_flat_mask[:, :3] = 1
        if hand_track:
            diff_local_body_pos_flat_mask[:, left_hand_id*3:(left_hand_id+1)*3] = 1
            diff_local_body_pos_flat_mask[:, right_hand_id*3:(right_hand_id+1)*3] = 1

        diff_local_body_rot_flat_mask = torch.zeros(1, 24 * 6)
        if not root_track and not hand_track:
            diff_local_body_rot_flat_mask[:, :6] = 1

        diff_local_vel_mask = torch.zeros(1, 24 * 3)
        if not root_track and not hand_track:
            diff_local_vel_mask[:, :3] = 1

        diff_local_ang_vel_mask = torch.zeros(1, 24 * 3)
        if not root_track and not hand_track:
            diff_local_ang_vel_mask[:, :3] = 1

        local_ref_body_pos_mask = torch.zeros(1, 24 * 3)
        local_ref_body_pos_mask[:, :3] = 1
        if hand_track:
            local_ref_body_pos_mask[:, left_hand_id*3:(left_hand_id+1)*3] = 1
            local_ref_body_pos_mask[:, right_hand_id*3:(right_hand_id+1)*3] = 1

        local_ref_body_rot_mask = torch.zeros(1, 24 * 6)
        if not root_track and not hand_track:
            local_ref_body_rot_mask[:, :6] = 1

        if include_action:
            noised_action_mask = torch.ones(1, self.action_size)
            relative_pose_dropout_mask = torch.cat([noised_action_mask, state_mask, diff_local_body_pos_flat_mask, diff_local_body_rot_flat_mask, diff_local_vel_mask,
                                                        diff_local_ang_vel_mask, local_ref_body_pos_mask, local_ref_body_rot_mask], dim=-1)
            assert relative_pose_dropout_mask.shape[-1] == self.input_size + self.action_size
        else:
            relative_pose_dropout_mask = torch.cat([state_mask, diff_local_body_pos_flat_mask, diff_local_body_rot_flat_mask, diff_local_vel_mask,
                                                        diff_local_ang_vel_mask, local_ref_body_pos_mask, local_ref_body_rot_mask], dim=-1)
            assert relative_pose_dropout_mask.shape[-1] == self.input_size
        return relative_pose_dropout_mask

    def sample_ts(self, B, device):
        if self.sample_t_strategy == "uniform":
            return torch.rand(B, device=device)
        elif self.sample_t_strategy == "lognormal":
            rnd_normal = torch.randn((B,), device=device)
            sigma = (rnd_normal * self.p_std + self.p_mean).exp()
            time = 1 / (1 + sigma)
            time = torch.clip(time, min=0.0001, max=1.0)
            return time

    def sample_actions(self, obs):
        assert not torch.is_grad_enabled(), "Autograd should not be enabled during the sampling chain!"

        B = obs.shape[0]
        x_0 = self.sample_noise([B, self.action_size], obs.device)
        time_grid = torch.tensor([0.0, 1.0], device=obs.device)

        if self.condition_drop_ratio > 0:
            condition_drop_ratio = self.relative_pose_dropout_mask.to(obs.device)
            batch_bernoulli_mask = self.sample_mask.to(obs.device)
            active_condition_mask = batch_bernoulli_mask * condition_drop_ratio + (1 - batch_bernoulli_mask) * torch.ones_like(condition_drop_ratio)
        else:
            active_condition_mask = None

        def velocity_fn(x, t, obs, condition_mask=None):
            obs_pointer = self.obs_norm(obs)
            x_eff = torch.zeros_like(x) if self.zero_action_input else x
            x_inp = torch.cat([x_eff, obs_pointer], dim=1)
            if condition_mask is not None:
                x_inp = x_inp * condition_mask

            t_batch = torch.ones([B], device=obs.device) * t
            noise_emb = self.noise_emb(t_batch * (0.0 if self.zero_action_input else 1.0))
            hidden = self.actor_mlp(x_inp)
            hidden = self.actor_norm(hidden, noise_emb)
            hidden = self.post_adaln_non_linearity(hidden)

            if self.parameterization == "velocity":
                velocity = self.mu(hidden)
            elif self.parameterization == "data":
                x1 = self.mu(hidden)
                velocity = self.path.target_to_velocity(x_1=x1, x_t=x, t=t_batch.unsqueeze(-1))
            return velocity

        x_1 = self.solver.sample(
            velocity_fn,
            time_grid=time_grid,
            x_init=x_0,
            method="euler",
            return_intermediates=False,
            atol=1e-5,
            rtol=1e-5,
            step_size=self.solver_step_size,
            obs=obs,
            condition_mask=active_condition_mask,
        )

        if self.perturb_action_std > 0:
            std = self.perturb_action_std
            std = torch.clamp(std, max=1e-6)
            x_1 = x_1 + torch.randn_like(x_1) * std

        return x_1

    def flow_matching_loss(self, actions, observations, t=None, noise=None, return_noise_t=False, sample_mask=None):
        noise = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        t = self.sample_ts(actions.shape[0], actions.device) if t is None else t
        path_sample = self.path.sample(t=t, x_0=noise, x_1=actions)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        if self.condition_drop_ratio > 0:
            batch_bernoulli_mask = sample_mask.to(observations.device)
            relative_pose_dropout_mask = self.relative_pose_dropout_mask.to(observations.device)
            active_condition_mask = batch_bernoulli_mask * relative_pose_dropout_mask + (1 - batch_bernoulli_mask) * torch.ones_like(relative_pose_dropout_mask)
        else:
            active_condition_mask = None

        with torch.cuda.amp.autocast():
            obs_pointer = self.obs_norm(observations)
            x_t_eff = torch.zeros_like(x_t) if self.zero_action_input else x_t
            x_inp = torch.cat([x_t_eff, obs_pointer], dim=-1)

            if active_condition_mask is not None:
                x_inp = x_inp * active_condition_mask.unsqueeze(1)
            
            noise_emb = self.noise_emb(t * (0.0 if self.zero_action_input else 1.0))
            hidden = self.actor_mlp(x_inp)
            hidden = self.actor_norm(hidden, noise_emb.unsqueeze(1))
            hidden = self.post_adaln_non_linearity(hidden)
            if self.parameterization == "velocity":
                velocity = self.mu(hidden)
                x1 = self.path.velocity_to_target(x_t=x_t, velocity=velocity, t=t.unsqueeze(-1).unsqueeze(-1))
                log_probs = -((u_t - velocity) ** 2) / (2 * 0.05 ** 2)
                loss = - log_probs.reshape(-1).mean()
            elif self.parameterization == "data":
                x1 = self.mu(hidden)
                log_probs = -((x1 - actions) ** 2) / (2 * 0.05 ** 2)
                loss = - log_probs.reshape(-1).mean()

        if return_noise_t:
            return log_probs.mean(-1).reshape(-1), loss, noise, t
        else:
            value = self.critic_mlp(obs_pointer)
            return log_probs.mean(-1).reshape(-1), loss, value

    def forward(self, observations):
        self.obs_pointer = self.obs_norm(observations)
        value = self.critic_mlp(self.obs_pointer)
        with torch.no_grad():
            actions = self.sample_actions(observations)
        return actions, value

    def _eval_mode(self):
        self.actor_mlp.eval()
        self.actor_norm.eval()
        self.mu.eval()
        self.noise_emb.eval()
        self.critic_mlp.eval()

    def _train_mode(self):
        self.actor_mlp.train()
        self.actor_norm.train()
        self.mu.train()
        self.noise_emb.train()
        self.critic_mlp.train()

    def learn(self):
        obs_dict = self.env.reset_all()
        obs_dict = {k: v.to(self.device) for k, v in obs_dict.items()}

        for it in range(self.current_learning_iteration, self.current_learning_iteration + self.config['num_learning_iterations']):
            self.current_learning_iteration = it

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            if it % self.config['save_interval'] == 0:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))
        self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.onnx"))

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.config['num_steps_per_env']):
                obs = torch.cat([obs_dict[k] for k in ['obs']], dim=1)
                actions = self.sample_actions(obs)
                value = self.critic_mlp(self.obs_norm(obs))
                obs_dict, rewards, dones, extras = self.env.step(actions)
                self.storage.add("obs", obs)
                self.storage.add("actions", actions)
                self.storage.add("rewards", rewards)
                self.storage.add("dones", dones)
                self.storage.add("values", value)
                obs_dict = {k: v.to(self.device) for k, v in obs_dict.items()}

            last_obs = torch.cat([obs_dict[k] for k in ['obs']], dim=1)
            last_values = self.critic_mlp(self.obs_norm(last_obs)).detach()
            returns, advantages = self._compute_returns_and_advantages(
                last_values,
                self.storage["values"].to(self.device),
                self.storage["dones"].to(self.device),
                self.storage["rewards"].to(self.device),
            )

            self.storage["returns"] = returns
            self.storage["advantages"] = advantages

        return obs_dict

    def _compute_returns_and_advantages(self, last_values, values, dones, rewards):
        advantage = 0
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
                next_is_not_terminal = 1.0 - dones[step].float()
                delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            else:
                next_values = values[step + 1]
                next_is_not_terminal = 1.0 - dones[step].float()
                delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _training_step(self):
        generator = self.storage.mini_batch_generator(self.config['num_mini_batches'], self.config['num_learning_epochs'])
        loss_dict = {"Flow": 0.0, "Value": 0.0}
        for minibatch in generator:
            loss_dict = self._update_algo_step(minibatch, loss_dict)

        num_updates = self.config['num_learning_epochs'] * self.config['num_mini_batches']
        for key in loss_dict:
            loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict

    def _update_algo_step(self, minibatch, loss_dict):
        fpo_loss_dict = self._compute_fpo_loss(minibatch)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss = fpo_loss_dict["actor_loss"] + fpo_loss_dict["critic_loss"]
        loss.backward()

        nn.utils.clip_grad_norm_(list(self.actor_mlp.parameters()) + list(self.actor_norm.parameters()) + list(self.mu.parameters()) + list(self.noise_emb.parameters()), self.config['max_grad_norm'])
        nn.utils.clip_grad_norm_(self.critic_mlp.parameters(), self.config['max_grad_norm'])

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict["Flow"] += fpo_loss_dict.pop("flow_loss").item()
        loss_dict["Value"] += fpo_loss_dict.pop("value_loss").item()
        return loss_dict

    def _compute_fpo_loss(self, minibatch):
        actions = minibatch["actions"]
        obs = minibatch["obs"]
        returns = minibatch["returns"]
        log_probs, flow_loss, value = self.flow_matching_loss(actions, obs)
        value_loss = (value - returns).pow(2).mean()
        actor_loss = flow_loss
        critic_loss = self.config['value_loss_coef'] * value_loss
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "flow_loss": flow_loss,
            "value_loss": value_loss,
        }

    def save(self, path, infos=None):
        checkpoint_dict = {
            "actor_mlp_state_dict": self.actor_mlp.state_dict(),
            "actor_norm_state_dict": self.actor_norm.state_dict(),
            "mu_state_dict": self.mu.state_dict(),
            "noise_emb_state_dict": self.noise_emb.state_dict(),
            "critic_mlp_state_dict": self.critic_mlp.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)

    def export(self, onnx_file_path):
        was_training = self.actor_mlp.training
        self._eval_mode()
        # For simplicity, skip full export
        if was_training:
            self._train_mode()

    def get_inference_policy(self, device=None):
        self.actor_mlp.eval()
        self.actor_norm.eval()
        self.mu.eval()
        self.noise_emb.eval()
        self.critic_mlp.eval()
        if device is not None:
            self.actor_mlp.to(device)
            self.actor_norm.to(device)
            self.mu.to(device)
            self.noise_emb.to(device)
            self.critic_mlp.to(device)

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            obs_tensor = obs['obs']
            return self.sample_actions(obs_tensor)

        return policy_fn