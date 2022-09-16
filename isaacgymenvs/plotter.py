import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Process

# To plot
# - (x,y) position
# - velocity of each wheel (expected vs. actual)
# - 

class Plotter:
    def __init__(self, dt, params):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.params = params
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def dump_states(self, dict):
        for key, value in dict.items():
            self.state_log[key] = list(value)

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log

        fig.suptitle(f"{self.params.title}", fontsize=14)
        # plot joint targets and measured positions
        a = axs[0,0]
        if log["x_pos"] and log["y_pos"]: a.plot(log["x_pos"], log["y_pos"], label='path')
        a.set(xlabel='x', ylabel='y', title=f'Crawler Path')
        a.legend()
        # plot left front wheel velocity
        a = axs[0,1]
        if log["track_lin_vel"]: a.plot(time, log["track_lin_vel"], label='measured')
        if log["cmd_lin_vel"]: a.plot(time, log["cmd_lin_vel"], label='target')
        a.set(xlabel='time [frames]', ylabel=' Linear Velocity (m/s)', title='Linear Velocity')
        a.legend()
        # plot right front wheel velocity
        a = axs[0,2]
        if log["track_ang_vel"]: a.plot(time, log["track_ang_vel"], label='measured')
        if log["cmd_ang_vel"]: a.plot(time, log["cmd_ang_vel"], label='target')
        a.set(xlabel='time [frames]', ylabel=' Angular Velocity [rad/s]', title='Angular Velocity')
        a.legend()
        a = axs[1,0]
        if log["l_wheel"]: a.plot(time, log["l_wheel"], label="l_wheel")
        if log["r_wheel"]: a.plot(time, log["r_wheel"], label="r_wheel")
        a.set(xlabel='time [frames]', ylabel=' Angular Velocity [rad/s]', title='Wheel Actions')
        a.legend()
        # plot left front wheel torque
        a = axs[1,1]
        if log["track_lin_vel"] and log["cmd_lin_vel"]: a.plot(time, np.sqrt((np.array(log["track_lin_vel"])-np.array(log["cmd_lin_vel"]))**2), label='measured')
        a.set(xlabel='time [frames]', ylabel='Error', title='Linear Velocity Error')
        a.legend()
        # plot right front wheel torque
        a = axs[1,2]
        if log["track_ang_vel"] and log["cmd_ang_vel"]: a.plot(time, np.sqrt(np.array((log["track_ang_vel"])-np.array(log["cmd_ang_vel"]))**2), label='measured')
        a.set(xlabel='time [frames]', ylabel='Error', title='Angular Velocity Error')
        a.legend()
        plt.show()

    def plot_eval(self):
        # self.plot_process = Process(target=self._plot_eval)
        # self.plot_process.start()
        self._plot_eval()

    def _plot_eval(self):
        nb_rows = 2
        nb_cols = 1
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        
        fig.suptitle(f"{self.params['title']}", fontsize=14)
        a = axs[0]
        if log["measured_lin_vel_mean"] and log["measured_lin_vel_std"] and log["target_lin_vel"]: 
            a.errorbar(log["target_lin_vel"], log["measured_lin_vel_mean"], yerr=log["measured_lin_vel_std"], fmt="o", color="blue", label="measured")
            a.plot(log["target_lin_vel"], log["target_lin_vel"], color="orange", label="target")
        a.set(xlabel='Target Linear Velocity', ylabel='Measured Linear Velocity')
        a.legend()
        a = axs[1]
        if log["measured_ang_vel_mean"] and log["measured_ang_vel_std"] and log["target_ang_vel"]: 
            a.errorbar(log["target_ang_vel"], log["measured_ang_vel_mean"], yerr=log["measured_ang_vel_std"], fmt="o", color="blue", label="measured")
            a.plot(log["target_ang_vel"], log["target_ang_vel"], color="orange", label="target")
        a.set(xlabel='Target Angular Velocity', ylabel='Measured Angular Velocity')
        a.legend()
        plt.savefig('fig.png')
        plt.show()

    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()


