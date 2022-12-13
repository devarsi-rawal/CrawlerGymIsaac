import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import to_torch, quat_rotate, quat_rotate_inverse, normalize, get_euler_xyz, get_axis_params
from .base.vec_task import VecTask

LIN_VEL_SIGMA = 0.01
ANG_VEL_SIGMA = 0.3

class Turtlebot(VecTask):
    lw, rw = "wheel_left_link", "wheel_right_link"
    # cw, cwb = "caster_wheel", "caster_wheel_base"
    lwj, rwj = "wheel_left_joint", "wheel_right_joint" 
    # cwj, cwbj = cw + "_joint", cwb + "_joint"

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = 2000  

        self.cfg["env"]["numObservations"] = 2 
        self.cfg["env"]["numActions"] = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.gym.viewer_camera_look_at(
                self.viewer, None, gymapi.Vec3(5, 5, 3), gymapi.Vec3(0, 0, 0))

        # New tensor inits
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 2, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, self.num_dof, 2)
        
        self.root_states = vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[:, 0:3]
        self.root_quats = self.root_states[:, 3:7]
        self.root_linvels = self.root_states[:, 7:10]
        self.root_angvels = self.root_states[:, 10:13]

        self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.marker_states = vec_root_tensor[:, 1, :]
        self.marker_positions = self.marker_states[:, 0:3]
        
        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_vels = vec_dof_tensor[..., 1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()
        
        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))
    
        # # Old tensor inits
        # dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # self.init_dof_state = self.dof_state.clone()
        # self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        # self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        #
        # self.actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # self.root_states = gymtorch.wrap_tensor(self.actor_root_state).view(self.num_envs, 2, 13)
        # self.tb_root_state = self.root_states[:, 0, :]
        # self.tb_init_root_state = self.tb_root_state.clone()
        # self.tb_base_pos = self.tb_root_state[:, 0:3]
        # self.tb_base_quat = self.tb_root_state[:, 3:7]
        # self.tb_base_euler = get_euler_xyz(self.tb_base_quat)
        # self.tb_base_lin_vel = quat_rotate_inverse(self.tb_base_quat, self.tb_root_state[:, 7:10])
        # self.tb_base_ang_vel = quat_rotate_inverse(self.tb_base_quat, self.tb_root_state[:, 10:13])
        
        self.direction_vector = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg["env"]["numActions"], 3), device=self.device) # Store up to 3 previous actions
    
        # # Config values
        # self.add_noise = self.cfg["env"]["addNoise"]
        # self.lin_vel_noise = self.cfg["env"]["linVelNoise"]
        # self.ang_vel_noise = self.cfg["env"]["angVelNoise"]
        # self.add_bias = self.cfg["env"]["addBias"]
        # self.lin_vel_bias = self.cfg["env"]["linVelBias"]
        # self.ang_vel_bias = self.cfg["env"]["angVelBias"]
        # self.bias_period = self.cfg["env"]["biasPeriod"]
        # self.lin_vel_var = self.cfg["env"]["linVelVariance"]
        # self.ang_vel_var = self.cfg["env"]["angVelVariance"]

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_ground_plane()
        num_envs_per_row = int(np.sqrt(self.num_envs))
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], num_envs_per_row)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/turtlebot.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.slices_per_cylinder = 50
        asset_options.disable_gravity = False
        # asset_options.armature = 0.001
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(asset)
        dof_names = self.gym.get_asset_dof_names(asset)

        marker_options = gymapi.AssetOptions()
        marker_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, 0.1, marker_options)

        print("== BODY INFORMATION ==")
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        print('num_bodies', self.num_bodies)
        body_dict = self.gym.get_asset_rigid_body_dict(asset)
        marker_body_dict = self.gym.get_asset_rigid_body_dict(marker_asset)
        print('rigid bodies', body_dict)
        print('marker body', marker_body_dict)
        dof_dict = self.gym.get_asset_dof_dict(asset)
        print('asset_dof', body_dict)
        dof_names = self.gym.get_asset_dof_names(asset)
        print('dof_names', dof_names)
    

        pose = gymapi.Transform()
        pose.p.z = 0.05

        self.envs = []
        self.actor_handles = []
        for i in range(self.num_envs):
            # create env instance

            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            actor_handle = self.gym.create_actor(env_ptr, asset, pose, "turtlebot", i, 1, 0)

            actor_dof_dict = self.gym.get_actor_dof_dict(env_ptr, actor_handle)
            self.wheel_dof = [actor_dof_dict[self.lwj], actor_dof_dict[self.rwj]]

            dof_props = self.gym.get_actor_dof_properties(env_ptr, actor_handle)
            dof_props['driveMode'][self.wheel_dof] = gymapi.DOF_MODE_VEL
            dof_props['friction'].fill(0.0)
            dof_props['stiffness'].fill(0.0)
            dof_props['stiffness'][self.wheel_dof] = 800.0
            dof_props['damping'].fill(0.0)
            dof_props['damping'][self.wheel_dof] = 200.0 
            dof_props['armature'].fill(0.0)
            dof_props['effort'].fill(1000.0)
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, dof_props)

            marker_pose = gymapi.Transform()
            marker_handle = self.gym.create_actor(env_ptr, marker_asset, marker_pose, "marker", i, 1, 1)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))

            self.envs.append(env_ptr)
            self.actor_handles.append(actor_handle)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)

        self.target_root_positions[env_ids, 0:2] = (torch.rand(num_sets, 2, device=self.device) * 6) - 3
        self.marker_positions[env_ids] = self.target_root_positions[env_ids]

        actor_indices = self.all_actor_indices[env_ids, 1].flatten()

        return actor_indices

    def reset_idx(self, env_ids):

        self.dof_vels = 0

        num_resets = len(env_ids)

        target_actor_indices = self.set_targets(env_ids)
        actor_indices = self.all_actor_indices[env_ids, 0].flatten()

        self.root_states[env_ids] = self.initial_root_states[env_ids]

        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        return torch.unique(torch.cat([target_actor_indices, actor_indices]))

    def pre_physics_step(self, _actions):

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        target_actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(set_target_ids) > 0:
            target_actor_indices = self.set_targets(set_target_ids)
            
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(reset_env_ids) > 0:
            actor_indices = self.reset_idx(reset_env_ids)

        reset_indices = torch.unique(torch.cat([target_actor_indices, actor_indices]))
        if len(reset_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(reset_indices), len(reset_indices))

        actions = _actions.to(self.device)
        self.prev_actions = torch.cat((torch.reshape(actions, (self.num_envs, self.cfg["env"]["numActions"], 1)), self.prev_actions), 2)[:, :, :3]
        linvel_action_scale = 0.5 # Max linear speed of TB
        angvel_action_scale = 1.0 # Max angular speed of TB
        v = actions[:, 0] * linvel_action_scale
        w = actions[:, 1] * angvel_action_scale

        lw_vel = (v - 0.23/2 * w) / 0.0352
        rw_vel = (v + 0.23/2 * w) / 0.0352

        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[self.wheel_dof[0]::self.num_dof] = lw_vel 
        actions_tensor[self.wheel_dof[1]::self.num_dof] = rw_vel 

        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor))

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        root_euler = get_euler_xyz(self.root_quats)
        self.direction_vector[:, 0] = torch.cos(root_euler[2] - np.pi/2) # Vector x-value
        self.direction_vector[:, 1] = torch.sin(root_euler[2] - np.pi/2) # Vector y-value
        self.target_vector = self.target_root_positions - self.root_positions
        self.local_target_vector = quat_rotate_inverse(self.root_quats, self.target_vector)
        self.distance = torch.linalg.norm(self.target_vector[:, 0:2], dim=1)
        self.heading_diff = torch.atan2(self.target_vector[:,0] * self.direction_vector[:,1] - self.target_vector[:,1]*self.direction_vector[:,0],
                                        self.target_vector[:,0]*self.direction_vector[:,0]+self.target_vector[:,1]*self.direction_vector[:,1])

        # Distance vector in dx and dy components
        self.obs_buf[..., :2] = self.local_target_vector[:, :2]
        # self.obs_buf[..., 0] = self.distance 
        # self.obs_buf[..., 1] = torch.cos(self.heading_diff)
        # self.obs_buf[..., 2] = torch.sin(self.heading_diff)

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_turtlebot_reward(
                    self.distance, self.prev_actions, self.reset_buf, self.progress_buf, self.max_episode_length
                )

@torch.jit.script
def compute_turtlebot_reward(distance, prev_actions, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor]

    # Minimize distance to goal
    distance_reward = 1.0 / (1.0 + (distance/0.1)**2) 
    distance_reward_scale = 10
    distance_reward *= distance_reward_scale
    # distance_reward = torch.where(distance < 0.03, max_episode_length - progress_buf, distance_reward)

    # Heading constraint
    lin_vel = prev_actions[:, 0, 0]
    heading_constraint_penalty_scale = -1 
    heading_constraint_penalty = torch.where(lin_vel < 0.0, 1, 0)
    heading_constraint_penalty *= heading_constraint_penalty_scale

    # Penalize high velocity near goal
    approach_vel_penalty_scale = -1
    approach_vel_penalty = (10 * prev_actions[:, 0, 0]**2) / (1 + (5 * distance)**2)
    approach_vel_penalty *= approach_vel_penalty_scale 


    # Penalize motion when reached goal
    motion_on_goal_penalty_scale = -5
    motion_on_goal_penalty = torch.where(distance < 0.03, torch.linalg.norm(prev_actions[..., 0], dim=1) / 0.5, torch.zeros_like(reset_buf))
    motion_on_goal_penalty *= motion_on_goal_penalty_scale
    
    reward = distance_reward + heading_constraint_penalty # + approach_vel_penalty + motion_on_goal_penalty

    # Define resets
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(distance > 6.0, ones, die) # Reset when tb goes 5m away from target
    
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die) # Reset tb that runs for more than max_episode_length

    return reward, reset
