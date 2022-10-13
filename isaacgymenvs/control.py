import datetime
import isaacgym
import time

import os
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import gym
import numpy as np

import onnx
import onnxruntime as ort

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

from isaacgymenvs.utils.utils import set_np_formatting, set_seed

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs
    import torch
    import pandas as pd
    from plotter import Plotter

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    # sets seed. if seed is -1 will pick a random one
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    # if cfg.wandb_activate and rank == 0:
    #     # Make sure to install WandB if you actually use this.
    #     import wandb
    #
    #     run = wandb.init(
    #         project=cfg.wandb_project,
    #         group=cfg.wandb_group,
    #         entity=cfg.wandb_entity,
    #         config=cfg_dict,
    #         sync_tensorboard=True,
    #         name=run_name,
    #         resume="allow",
    #         monitor_gym=True,
    #     )

    cfg.task_name = "CrawlerControl"

    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    # # register the rl-games adapter to use inside the runner
    # vecenv.register('RLGPU',
    #                 lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    # env_configurations.register('rlgpu', {
    #     'vecenv_type': 'RLGPU',
    #     'env_creator': create_env_thunk,
    # })
    #
    # # register new AMP network builder and agent
    # def build_runner(algo_observer):
    #     runner = Runner(algo_observer)
    #     runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
    #     runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
    #     model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
    #     model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
    #
    #     return runner
    #
    # rlg_config_dict = omegaconf_to_dict(cfg.train)
    #
    # # convert CLI arguments into dictionory
    # # create runner and set the settings
    # runner = build_runner(RLGPUAlgoObserver())
    # runner.load(rlg_config_dict)
    # runner.reset()
    #
    # # dump config dict
    # experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    # os.makedirs(experiment_dir, exist_ok=True)
    # with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
    #     f.write(OmegaConf.to_yaml(cfg))
    print(cfg.onnx_model)
    fn = "runs/crawler-export/nn/crawler-export.onnx"
    onnx_model = onnx.load(fn)
    ort_model = ort.InferenceSession(fn)

    env = create_env_thunk()
    obs_dict = env.reset()
    obs = obs_dict["obs"].detach().cpu().numpy().reshape((1, -1))
    print("obs:", obs) 
    outputs = ort_model.run(None, {"obs": obs})
    print("outputs:", outputs)
    _actions = outputs[0]
    print("actions:", _actions)
    num_steps = 0
    total_reward = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        obs = obs_dict["obs"].detach().cpu().numpy().reshape((1, -1))
        outputs = ort_model.run(None, {"obs": obs})
        _actions = outputs[0]
        print(_actions)
        actions = torch.from_numpy(_actions).to(cfg.sim_device)
        obs_dict, reward, reset, info = env.step(actions)
        num_steps += 1
        env.render()


        

if __name__ == "__main__":
    launch_rlg_hydra()
