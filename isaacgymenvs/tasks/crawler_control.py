import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import quat_rotate_inverse, normalize
from .base.vec_task import VecTask

class CrawlerControl(VecTask):
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
        #     lin = np.linspace(-0.2, 0.2, num=self.cfg["eval"]["linVelStep"])
        #     ang = np.linspace(-1, 1, num=self.cfg["eval"]["angVelStep"])
        #     h = np.linspace(0, 2*np.pi, num=self.cfg["eval"]["headingStep"])
        #     s = np.linspace(0, 2*np.pi, num=self.cfg["eval"]["swivelStep"])
        #     self.eval_params = (np.array(np.meshgrid(lin, ang, h, s)).T).reshape(-1,4)
        #     cmds = self.eval_params[:, :2]
        #     self.headings = self.eval_params[:, 2]
        #     self.swivels = self.eval_params[:, 3]
        self.init_heading = self.cfg["env"]["initHeading"]
        self.init_swivel = self.cfg["env"]["initSwivel"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        self.gym.viewer_camera_look_at(
                self.viewer, None, gymapi.Vec3(5, 5, 3), gymapi.Vec3(0, 0, 0))
    
        self.turn = 0.0 
        self.forward = 0.0 
        self.lin_vel = 0.0 
        self.ang_vel = 0.0

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "inc_lin_vel")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "dec_lin_vel")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "inc_ang_vel")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "dec_ang_vel")
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        _rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_tensor)
        self.rb_positions = rb_states[:, 0:3].view(self.num_envs, self.num_bodies, 3)
        self.rb_orients = rb_states[:, 3:7].view(self.num_envs, self.num_bodies, 4)

        self.tether_base = torch.tensor([0, 0., 0], dtype=torch.float, requires_grad=False, device=self.device_id).repeat(self.num_envs, self.num_bodies, 1)
        
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

         
        # init_dof_pos = self.dof_state.clone() 
        # if self.evaluate:
        #     init_dof_pos[self.cwb_dof::self.num_dof, 0] = torch.ones(self.num_envs, dtype=torch.float, device=self.device_id, requires_grad=False) * self.cfg["env"]["swivelPosition"]
        # else:
        #     init_dof_pos[self.cwb_dof::self.num_dof, 0] = torch.rand(self.num_envs, dtype=torch.float, device=self.device_id, requires_grad=False) * 2 * np.pi 
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(init_dof_pos))

        # TODO: Add 2 (# of commands) to cfg task file
        self.commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device_id, requires_grad=False) 
        # self._resample_commands()

        # if self.evaluate:
        #     self.set_commands(cmds)
        #     self.set_swivel_pos(self.swivels)
        # else:
        #     init_dof_pos = self.dof_state.clone() 
        #     init_dof_pos[self.cwb_dof::self.num_dof, 0] = torch.randn(self.num_envs, dtype=torch.float, device=self.device_id, requires_grad=False) * 2 * np.pi 
        #     self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(init_dof_pos))
        s = np.zeros(self.num_envs) + self.init_swivel
        self.set_swivel_pos(s)

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.vertical = self.cfg["env"]["vertical"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], self.num_envs)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        # plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0) if self.vertical else gymapi.Vec3(0.0, 0.0, 1.0)

        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, 0.0, -spacing) if self.vertical else gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, 0.0, spacing) if self.vertical else gymapi.Vec3(0.5 * spacing, spacing, spacing) 

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

        self.crawler_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            pose = gymapi.Transform()
            # if self.evaluate:
            #     heading = self.headings[i]
            # else:
            #     heading = np.random.rand() * 2 * np.pi
            heading = self.init_heading

            if self.vertical:
                pose.p.y = 0.05
                pose.r = gymapi.Quat.from_euler_zyx(-np.pi/2, heading, 0)
            else:
                pose.p.z = 0.05
                pose.r = gymapi.Quat.from_euler_zyx(heading, 0, 0)
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

        self.obs_buf[env_ids, 0] = self.base_lin_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.base_ang_vel[env_ids, 2].squeeze()
        self.obs_buf[env_ids, 2] = self.commands[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 3] = self.commands[env_ids, 1].squeeze()
        
        if self.add_noise:
            self.obs_buf[env_ids, 0] += torch.randn(env_ids.size, dtype=torch.float, device=self.device_id, requires_grad=False) * (self.lin_vel_noise)
            self.obs_buf[env_ids, 1] += torch.randn(env_ids.size, dtype=torch.float, device=self.device_id, requires_grad=False) * (self.ang_vel_noise)
        if self.add_bias:
            self.obs_buf[env_ids, 0] += torch.sin(self.last_step*self.bias_period*torch.ones(env_ids.size, dtype=torch.float, device=self.device_id, requires_grad=False)) * self.lin_vel_bias
            self.obs_buf[env_ids, 1] += torch.sin(self.last_step*self.bias_period*torch.ones(env_ids.size, dtype=torch.float, device=self.device_id, requires_grad=False)) * self.ang_vel_bias

    def get_observations(self):
        return self.obs_buf

    def reset_idx(self, env_ids):
        pass
        # positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        # velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        # self.dof_pos[env_ids, :] = positions[:]
        # self.dof_vel[env_ids, :] = velocities[:]

        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # self.reset_buf[env_ids] = 0
        # self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # TODO: Add control dof numbers to cfg file
        # actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        # TODO: Add action_scale to task cfg file
        _actions = actions.to(self.device) * 10.0 
         
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[self.wheel_dof[0]::self.num_dof] = _actions[:,0]
        actions_tensor[self.wheel_dof[1]::self.num_dof] = _actions[:,1]
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor))

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self._apply_forces()

        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "left":
                print("TURNING LEFT")
                self.turn = 0.5 if evt.value > 0 else 0.5
            if evt.action == "right":
                self.turn = -0.5 if evt.value > 0 else -0.5
                print("TURNING RIGHT")
            if evt.action == "up":
                self.forward = 0.2 if evt.value > 0 else 0.2
                print("GOING FORWARD")
            if evt.action == "down":
                self.forward = -0.2 if evt.value > 0 else -0.2 
                print("GOING BACKWARD")
            # if evt.action == "inc_lin_vel":
            #     self.lin_vel += 0.02 if evt.value > 0 else 0
            # if evt.action == "dec_lin_vel":
            #     self.lin_vel -= 0.02 if evt.value > 0 else 0
            # if evt.action == "inc_ang_vel":
            #     self.ang_vel += 0.2 if evt.value > 0 else 0
            # if evt.action == "dec_ang_vel":
            #     self.ang_vel -= 0.2 if evt.value > 0 else 0

        print(f"turn: {self.turn}, forward: {self.forward}") # , lin_vel: {self.lin_vel}, ang_vel: {self.ang_vel}")
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        self.commands[:, 0] = torch.zeros((self.num_envs), device=self.device_id) + self.forward
        self.commands[:, 1] = torch.zeros((self.num_envs), device=self.device_id) + self.turn
        # print(self.commands)

        # if self.cfg["env"]["randomCommands"]:
        #     env_ids = (self.progress_buf % 250 == 0).nonzero(as_tuple=False).flatten() 
        #     self._resample_commands(env_ids)

    def _apply_forces(self):
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
            magnet_forces[:, 3, 0] = -torch.sin(theta_cw) * self.magnetism 
            magnet_forces[:, 3, 2] = torch.cos(theta_cw) * self.magnetism
            # self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE)
        else:
            force_dir = 1 if self.vertical else 2
            magnet_forces[:, 1, force_dir] = self.magnetism
            magnet_forces[:, 2, force_dir] = self.magnetism * 0.1
            # self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(self.rb_positions.clone()), gymapi.ENV_SPACE)

        # Apply tether force
        _tether_forces = self.tether * normalize(self.tether_base - self.rb_positions)
        tether_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float, requires_grad=False)
        tether_forces[:, 1, 2] = _tether_forces[:, 1, 2] 

        final_orients = self.rb_orients[:, 1, :][:, None, :].repeat(1, self.num_bodies, 1)
        # print("RB Orients Shape", self.rb_orients[:, 1, :].shape)
        # print("Final Orients Shape", final_orients.shape, final_orients[:, 1, :].shape)
        # print("Tether Forces Shape", _tether_forces.shape, _tether_forces[:, 1, :].shape)
        tether_forces[:, 1, :] = quat_rotate_inverse(final_orients[:, 1, :], _tether_forces[:, 1, :])
        forces = magnet_forces + tether_forces
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE)

    def set_commands(self, cmds):
        self.commands[:] = torch.from_numpy(cmds).float().to(self.device_id)

    def _resample_commands(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        # TODO: Add randomness to command resampling
        self.commands[env_ids, 0] = 2 * 0.2 * torch.rand(len(env_ids), device=self.device_id) - 0.2
        self.commands[env_ids, 1] = 2 * 1.0 * torch.rand(len(env_ids), device=self.device_id) - 1.0

    # def set_heading(self, headings):


    def set_swivel_pos(self, swivel_pos):
        init_dof_pos = self.dof_state.clone() 
        init_dof_pos[self.cwb_dof::self.num_dof, 0] = torch.tensor(swivel_pos, device=self.device_id, dtype=torch.float, requires_grad=False)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(init_dof_pos))

@torch.jit.script
def compute_crawler_reward(measured_lin_vel, measured_ang_vel, target_lin_vel, target_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

    lm = target_lin_vel != 0
    lin_vel_error = torch.empty_like(target_lin_vel)
    lin_vel_error[lm] = 1 - torch.abs((target_lin_vel[lm] - measured_lin_vel[lm])/target_lin_vel[lm])
    lin_vel_error[~lm] = 1 - torch.abs(target_lin_vel[~lm] - measured_lin_vel[~lm])
    am = target_ang_vel != 0
    ang_vel_error = torch.empty_like(target_ang_vel)
    ang_vel_error[am] = 1 - torch.abs((target_ang_vel[am] - measured_ang_vel[am])/target_ang_vel[am])
    ang_vel_error[~am] = 1 - torch.abs(target_ang_vel[~am] - measured_ang_vel[~am])

    reward = lin_vel_error + ang_vel_error

    return reward

@torch.jit.script
def compute_crawler_reward_exp(measured_lin_vel, measured_ang_vel, target_lin_vel, target_ang_vel, lin_vel_var, ang_vel_var):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    lin_vel_error = torch.square(target_lin_vel - measured_lin_vel)
    ang_vel_error = torch.square(target_ang_vel - measured_ang_vel)
    reward = torch.exp(-lin_vel_error/lin_vel_var) + torch.exp(-ang_vel_error/ang_vel_var)
    return reward
