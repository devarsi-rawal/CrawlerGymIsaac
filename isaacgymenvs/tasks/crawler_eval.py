import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import to_torch, quat_rotate, quat_rotate_inverse, normalize, get_euler_xyz, get_axis_params
from .base.vec_task import VecTask

LIN_VEL_SIGMA = 0.01
ANG_VEL_SIGMA = 0.3

class CrawlerEval(VecTask):
    lw, rw = "left_front_wheel", "right_front_wheel"
    cw, cwb = "caster_wheel", "caster_wheel_base"
    lwj, rwj = lw + "_joint", rw + "_joint"
    cwj, cwbj = cw + "_joint", cwb + "_joint"

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 2


        self.evaluate = self.cfg["eval"]["evaluate"]

        # if self.evaluate:
        #     h = np.linspace(0, 2*np.pi, num=self.cfg["eval"]["headingStep"])
        #     s = np.linspace(0, 2*np.pi, num=self.cfg["eval"]["swivelStep"])
        #     self.eval_params = (np.array(np.meshgrid(h, s)).T).reshape(-1, 2)
        #     self.headings = self.eval_params[:, 0]
        #     self.swivels = self.eval_params[:, 1]
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.gym.viewer_camera_look_at(
                self.viewer, None, gymapi.Vec3(5, 5, 3), gymapi.Vec3(0, 0, 0))

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        _rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_tensor)
        self.rb_positions = rb_states[:, 0:3].view(self.num_envs, self.num_bodies, 3)
        self.rb_orients = rb_states[:, 3:7].view(self.num_envs, self.num_bodies, 4)

        self.tether_base = torch.tensor([0, 0., 0], dtype=torch.float, requires_grad=False, device=self.device_id).repeat(self.num_envs, self.num_bodies, 1)

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.magnetism = -self.cfg["env"]["magnetism"]
        self.tether = self.cfg["env"]["tether"]
        self.add_noise = self.cfg["env"]["addNoise"]
        self.lin_vel_noise = self.cfg["env"]["linVelNoise"]
        self.ang_vel_noise = self.cfg["env"]["angVelNoise"]
        self.add_bias = self.cfg["env"]["addBias"]
        self.lin_vel_bias = self.cfg["env"]["linVelBias"]
        self.ang_vel_bias = self.cfg["env"]["angVelBias"]
        self.bias_period = self.cfg["env"]["biasPeriod"]
        self.lin_vel_var = self.cfg["env"]["linVelVariance"]
        self.ang_vel_var = self.cfg["env"]["angVelVariance"]

        # TODO: Add 2 (# of commands) to cfg task file
        self.commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device_id, requires_grad=False) 
        self._resample_commands()

        self.initial_dof_states = self.dof_state.clone()

        self.initial_root_states = self.root_states.clone()

        # print("Init DOF States: ", self.initial_dof_states)
        # print("Init Root States: ", self.initial_root_states)

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis_idx = 2
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.vertical = self.cfg["env"]["vertical"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_ground_plane()
        num_envs_per_row = self.num_envs if self.vertical else int(np.sqrt(self.num_envs))
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], num_envs_per_row)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        # plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0) if self.vertical else gymapi.Vec3(0.0, 0.0, 1.0)

        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, 0.0, -spacing) if self.vertical else gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, 0.0, spacing) if self.vertical else gymapi.Vec3(spacing, spacing, spacing) 

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/crawler/crawler_caster.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.replace_cylinder_with_capsule = True
        asset_options.disable_gravity = False
        asset_options.armature = 0.001
        crawler_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(crawler_asset)
        dof_names = self.gym.get_asset_dof_names(crawler_asset)

        print("== BODY INFORMATION ==")
        self.num_bodies = self.gym.get_asset_rigid_body_count(crawler_asset)
        print('num_bodies', self.num_bodies)
        body_dict = self.gym.get_asset_rigid_body_dict(crawler_asset)
        print('rigid bodies', body_dict)
        dof_dict = self.gym.get_asset_dof_dict(crawler_asset)
        print('asset_dof', body_dict)
        dof_names = self.gym.get_asset_dof_names(crawler_asset)
        print('dof_names', dof_names)

        pose = gymapi.Transform()
        if self.vertical:
            pose.p.y = 0.05
            pose.r = gymapi.Quat.from_euler_zyx(-np.pi/2, -np.pi/2, 0)
        else:
            pose.p.z = 0.025
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.crawler_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            pose = gymapi.Transform()

            if self.vertical:
                pose.p.y = 0.05
                pose.r = gymapi.Quat.from_euler_zyx(-np.pi/2, np.pi, 0)
            else:
                pose.p.z = 0.025
                pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            crawler_handle = self.gym.create_actor(env_ptr, crawler_asset, pose, "crawler", i, 1, 0)

            actor_dof_dict = self.gym.get_actor_dof_dict(env_ptr, crawler_handle)
            self.wheel_dof = [actor_dof_dict[self.lwj], actor_dof_dict[self.rwj]]
            self.cw_dof = actor_dof_dict[self.cwj]
            self.cwb_dof = actor_dof_dict[self.cwbj]

            dof_props = self.gym.get_actor_dof_properties(env_ptr, crawler_handle)
            dof_props['driveMode'][self.wheel_dof] = gymapi.DOF_MODE_VEL
            dof_props['friction'].fill(0.01)
            dof_props['stiffness'].fill(0.0)
            dof_props['stiffness'][self.wheel_dof] = 800.0
            dof_props['damping'].fill(0.0)
            dof_props['damping'][self.wheel_dof] = 200.0 
            dof_props['armature'].fill(0.001)
            dof_props['effort'].fill(1000.0)
            self.gym.set_actor_dof_properties(env_ptr, crawler_handle, dof_props)

            self.envs.append(env_ptr)
            self.crawler_handles.append(crawler_handle)


    def compute_reward(self):
        # retrieve environment observations from buffer
        measured_lin_vel = self.obs_buf[:, 0]
        measured_ang_vel = self.obs_buf[:, 1]
        target_lin_vel = self.obs_buf[:, 2]
        target_ang_vel = self.obs_buf[:, 3]

        if self.cfg["env"]["expRew"]:
            self.rew_buf[:] = compute_crawler_reward_exp(
                measured_lin_vel, measured_ang_vel, target_lin_vel, target_ang_vel, self.lin_vel_var, self.ang_vel_var
            )
        else:
            self.rew_buf[:] = compute_crawler_reward(
                measured_lin_vel, measured_ang_vel, target_lin_vel, target_ang_vel
            )


    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        projected_gravity = quat_rotate(self.base_quat, self.gravity_vec)

        self.obs_buf[env_ids, 0] = self.base_lin_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.base_ang_vel[env_ids, 2].squeeze()
        self.obs_buf[env_ids, 2] = self.commands[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 3] = self.commands[env_ids, 1].squeeze()
        # self.obs_buf[env_ids, 4:] = projected_gravity 
        self.ideal_obs_buf = self.obs_buf.clone()
        if self.add_noise:
            self.obs_buf[env_ids, 0] += torch.randn(env_ids.size, dtype=torch.float, device=self.device_id, requires_grad=False) * (self.lin_vel_noise)
            self.obs_buf[env_ids, 1] += torch.randn(env_ids.size, dtype=torch.float, device=self.device_id, requires_grad=False) * (self.ang_vel_noise)
        if self.add_bias:
            self.obs_buf[env_ids, 0] += torch.sin(self.last_step*self.bias_period*torch.ones(env_ids.size, dtype=torch.float, device=self.device_id, requires_grad=False)) * self.lin_vel_bias
            self.obs_buf[env_ids, 1] += torch.sin(self.last_step*self.bias_period*torch.ones(env_ids.size, dtype=torch.float, device=self.device_id, requires_grad=False)) * self.ang_vel_bias

    def get_observations(self):
        return self.obs_buf

    def reset_idx(self, env_ids):
        # pass
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor(self.sim,
                                             gymtorch.unwrap_tensor(self.initial_root_states))
        self.gym.set_dof_state_tensor(self.sim,
                                      gymtorch.unwrap_tensor(self.initial_dof_states))

        # self.reset_buf[env_ids] = 0
        # self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # TODO: Add control dof numbers to cfg file
        # actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        # TODO: Add action_scale to task cfg file
        # 10.0 rad/sec wheel velocity
        _actions = actions.to(self.device) * 10.0 
         
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[self.wheel_dof[0]::self.num_dof] = _actions[:,0]
        actions_tensor[self.wheel_dof[1]::self.num_dof] = _actions[:,1]
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor))
        # import pdb
        # pdb.set_trace()

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.cfg["env"]["tetherApply"] == 0:
            self._apply_forces()
        elif self.cfg["env"]["tetherApply"] == 1:
            env_ids = (self.progress_buf % 100 == 0).nonzero(as_tuple=False).flatten() 
            self._apply_forces(env_ids)
        else:  
            self._apply_forces()

        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(env_ids) > 0:
        #     self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        if self.cfg["env"]["randomCommands"] and not self.evaluate:
            env_ids = (self.progress_buf % 250 == 0).nonzero(as_tuple=False).flatten() 
            self._resample_commands(env_ids)

    def _apply_forces(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        magnet_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device_id, dtype=torch.float, requires_grad=False)
        # apply magnetic force
        if self.cfg["env"]["localForce"]:
            theta_lw = self.dof_state[self.wheel_dof[0]::self.num_dof, 0]
            magnet_forces[:, 4, 0] = -torch.sin(theta_lw) * self.magnetism
            magnet_forces[:, 4, 2] = torch.cos(theta_lw) * self.magnetism
            theta_rw = self.dof_state[self.wheel_dof[1]::self.num_dof, 0]
            magnet_forces[:, 5, 0] = -torch.sin(theta_rw) * self.magnetism
            magnet_forces[:, 5, 2] = torch.cos(theta_rw) * self.magnetism
            theta_cw = self.dof_state[self.cw_dof::self.num_dof, 0]
            magnet_forces[:, 3, 0] = -torch.sin(theta_cw) * self.magnetism * 0.8 
            magnet_forces[:, 3, 2] = torch.cos(theta_cw) * self.magnetism * 0.8 
            # self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE)
        else:
            force_dir = 1 if self.vertical else 2
            magnet_forces[:, 1, force_dir] = self.magnetism
            magnet_forces[:, 2, force_dir] = self.magnetism * 0.1
            # self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(self.rb_positions.clone()), gymapi.ENV_SPACE)

        # Apply tether force
        _tether_forces = self.tether * normalize(self.tether_base - self.rb_positions)
        tether_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float, requires_grad=False)
        tether_forces[env_ids, 1, 1] = _tether_forces[env_ids, 1, 1] 

        final_orients = self.rb_orients[env_ids, 1, :][:, None, :].repeat(1, self.num_bodies, 1)
        # print("RB Orients Shape", self.rb_orients[:, 1, :].shape)
        # print("Final Orients Shape", final_orients.shape, final_orients[:, 1, :].shape)
        # print("Tether Forces Shape", _tether_forces.shape, _tether_forces[:, 1, :].shape)
        tether_forces[env_ids, 1, :] = quat_rotate_inverse(final_orients[env_ids, 1, :], tether_forces[env_ids, 1, :])
        # forces = magnet_forces + tether_forces
        if self.cfg["env"]["tetherApply"] == 2:
            forces = tether_forces * torch.sin(self.progress_buf)[0]
        else:
            forces = tether_forces
        # print(forces)
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE)

    def set_commands(self, cmds):
        self.commands[:] = torch.from_numpy(cmds).float().to(self.device_id)

    def _resample_commands(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        # TODO: Add randomness to command resampling
        self.commands[env_ids, 0] = 2 * 0.2 * torch.rand(len(env_ids), device=self.device_id) - 0.2
        self.commands[env_ids, 1] = 2 * 1.0 * torch.rand(len(env_ids), device=self.device_id) - 1.0

    def set_heading(self, headings):
        if self.vertical:
            orients = [gymapi.Quat.from_euler_zyx(-np.pi/2, heading, 0) for heading in headings]
            self.initial_root_states[:, 1] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device_id, requires_grad=False) + 0.05
            initial_orients = [[o.x, o.y, o.z, o.w] for o in orients]
            self.initial_root_states[:, 3:7] = to_torch(initial_orients, device=self.device, requires_grad=False) 
        else:
            orients = [gymapi.Quat.from_euler_zyx(0.0, 0.0, heading) for heading in headings]
            self.initial_root_states[:, 2] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device_id, requires_grad=False) + 0.05
            initial_orients = [[o.x, o.y, o.z, o.w] for o in orients]
            self.initial_root_states[:, 3:7] = to_torch(initial_orients, device=self.device, requires_grad=False) 
        self.gym.set_actor_root_state_tensor(self.sim,
                                             gymtorch.unwrap_tensor(self.initial_root_states))


    def set_swivel_pos(self, swivel_pos):
        self.initial_dof_states[self.cwb_dof::self.num_dof, 0] = torch.tensor(swivel_pos, device=self.device_id, dtype=torch.float, requires_grad=False)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_dof_states))

@torch.jit.script
def compute_crawler_reward_old(measured_lin_vel, measured_ang_vel, target_lin_vel, target_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

    lm = target_lin_vel != 0
    lin_vel_error = torch.empty_like(target_lin_vel)
    lin_vel_error[lm] = 1 - torch.abs((target_lin_vel[lm] - measured_lin_vel[lm]))/torch.abs(target_lin_vel[lm])
    lin_vel_error[~lm] = 1 - torch.abs(target_lin_vel[~lm] - measured_lin_vel[~lm])
    am = target_ang_vel != 0
    ang_vel_error = torch.empty_like(target_ang_vel)
    ang_vel_error[am] = 1 - torch.abs((target_ang_vel[am] - measured_ang_vel[am])/target_ang_vel[am])
    ang_vel_error[~am] = 1 - torch.abs(target_ang_vel[~am] - measured_ang_vel[~am])

    reward = lin_vel_error + ang_vel_error

    return reward

@torch.jit.script
def compute_crawler_reward(measured_lin_vel, measured_ang_vel, target_lin_vel, target_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    lin_vel_error = 1 - torch.abs(target_lin_vel - measured_lin_vel)
    ang_vel_error = 1 - torch.abs(target_ang_vel - measured_ang_vel)
    reward = lin_vel_error + ang_vel_error
    return reward

@torch.jit.script
def compute_crawler_reward_exp(measured_lin_vel, measured_ang_vel, target_lin_vel, target_ang_vel, lin_vel_var, ang_vel_var):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    lin_vel_error = torch.square(target_lin_vel - measured_lin_vel)
    ang_vel_error = torch.square(target_ang_vel - measured_ang_vel)
    lin_reward = 0.9 * torch.exp(-lin_vel_error/lin_vel_var) + 0.1 * torch.exp(-lin_vel_error/(100 * lin_vel_var))
    ang_reward = 0.9 * torch.exp(-ang_vel_error/ang_vel_var) + 0.1 * torch.exp(-ang_vel_error/(100 * ang_vel_var))
    reward = lin_reward * ang_reward
    return reward
