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

    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
            monitor_gym=True,
        )

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

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    def evaluate(player):
        plot_params = {
                "title": cfg.experiment 
        }
        plotter = Plotter(1, plot_params)
        cmds = np.stack((np.arange(-0.2, 0.2, 0.4/player.env.num_envs), np.arange(-1.0, 1.0, 2.0/player.env.num_envs)), axis=1)
        measured_lin_vels = np.empty((player.env.num_envs, 1), dtype=float) 
        measured_ang_vels = np.empty((player.env.num_envs, 1), dtype=float) 

        n_games = player.games_num
        render = player.render_env
        n_game_life = player.n_game_life
        is_determenistic = player.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(player.env, "has_action_mask", None) is not None

        op_agent = getattr(player.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = player.env.has_action_mask()

        need_init_rnn = player.is_rnn
        for _ in range(1):
            if games_played >= n_games:
                break

            player.env.set_commands(cmds)
            obses = player.env_reset(player.env)
            batch_size = 1
            batch_size = player.get_batch_size(obses, batch_size)

            if need_init_rnn:
                player.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(player.games_num):
                if has_masks:
                    masks = player.env.get_action_mask()
                    action = player.get_masked_action(
                        obses, masks, is_determenistic)
                else:
                    action = player.get_action(obses, is_determenistic)

                obses, r, done, info = player.env_step(player.env, action)
                print(obses[0])
                cr += r
                steps += 1

                measured_lin_vels = np.hstack((measured_lin_vels, obses[:,0].detach().cpu().numpy().reshape(-1,1)))
                measured_ang_vels = np.hstack((measured_ang_vels, obses[:,1].detach().cpu().numpy().reshape(-1,1)))

                if render:
                    player.env.render(mode='human')
                    time.sleep(player.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::player.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if player.is_rnn:
                        for s in player.states:
                            s[:, all_done_indices, :] = s[:,all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if player.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count,
                                  'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count,
                                  'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    # if batch_size//player.num_agents == 1 or games_played >= n_games:
                    #     break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

        mean_lin_vels = np.mean(measured_lin_vels, axis=1)
        std_lin_vels = np.std(measured_lin_vels, axis=1)
        mean_ang_vels = np.mean(measured_ang_vels, axis=1)
        std_ang_vels = np.std(measured_ang_vels, axis=1)
        

        logger_vars = {
                "measured_lin_vel_mean": mean_lin_vels, 
                "measured_lin_vel_std": std_lin_vels,
                "measured_ang_vel_mean": mean_ang_vels,
                "measured_ang_vel_std": std_ang_vels, 
                "target_lin_vel": cmds[:,0],
                "target_ang_vel": cmds[:,1]
                }
        plotter.dump_states(logger_vars)
                
        plotter.plot_eval()

    player = runner.create_player()
    if cfg.checkpoint is not None and cfg.checkpoint != '':
        player.restore(cfg.checkpoint)
    evaluate(player)

if __name__ == "__main__":
    launch_rlg_hydra()

