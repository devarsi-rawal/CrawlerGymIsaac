import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Process

# To plot
# - (x,y) position
# - velocity of each wheel (expected vs. actual)
# - 

LOGGER_PATH="data/"

class Plotter:
    def __init__(self, dt, params):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.params = params
        self.plot_process = None
        plt.rcParams["figure.figsize"] = (11,10)

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def dump_states(self, dict):
        for key, value in dict.items():
            self.state_log[key] = value

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

    def plot_error(self):
        self._plot_error_ind()

    def _plot_error(self):
        nb_rows = 2
        nb_cols = 2
        fig, axs = plt.subplots(nb_rows, nb_cols)
        log= self.state_log
        size = log["size"]
        ticks = np.arange(size) 
        
        fig.suptitle(", ".join([f"{k}: {v}" for k, v in self.params.items()]), fontsize=8)
        a = axs[0,0]
        mat = np.abs(log["measured_lin_vel_mean"] - log["target_lin_vel"]).reshape(-1, size)
        img = a.imshow(mat, cmap="jet")
        a.set_xticks(ticks)
        a.set_yticks(ticks)
        a.set_xticklabels(np.round(np.unique(log["target_lin_vel"]), 4), rotation=45)
        a.set_yticklabels(np.round(np.unique(log["target_ang_vel"]), 4))
        a.tick_params(axis='both', which='major', labelsize=6)
        plt.colorbar(img, ax=a)
        a.set_title("Linear Velocity Error")
        a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
        
        a = axs[1,0]
        mat = np.abs(log["measured_ang_vel_mean"] - log["target_ang_vel"]).reshape(-1, size)
        img = a.imshow(mat, cmap="jet")
        a.set_xticks(ticks)
        a.set_yticks(ticks)
        a.set_xticklabels(np.round(np.unique(log["target_lin_vel"]), 4), rotation=45)
        a.set_yticklabels(np.round(np.unique(log["target_ang_vel"]), 4))
        a.tick_params(axis='both', which='major', labelsize=6)
        plt.colorbar(img, ax=a)
        a.set_title("Angular Velocity Error")
        a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
        a = axs[0,1]
              
        img = a.imshow(log["measured_lin_vel_std"].reshape(-1, size), cmap="jet")
        a.set_xticks(ticks)
        a.set_yticks(ticks)
        a.set_xticklabels(np.round(np.unique(log["target_lin_vel"]), 4), rotation=45)
        a.set_yticklabels(np.round(np.unique(log["target_ang_vel"]), 4))
        a.tick_params(axis='both', which='major', labelsize=6)
        plt.colorbar(img, ax=a)
        a.set_title("Linear Velocity StDev")
        a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
        a = axs[1,1]

        img = a.imshow(log["measured_ang_vel_std"].reshape(-1, size), cmap="jet")
        a.set_xticks(ticks)
        a.set_yticks(ticks)
        a.set_xticklabels(np.round(np.unique(log["target_lin_vel"]), 4), rotation=45)
        a.set_yticklabels(np.round(np.unique(log["target_ang_vel"]), 4))
        a.tick_params(axis='both', which='major', labelsize=6)
        plt.colorbar(img, ax=a)
        a.set_title("Angular Velocity StDev")
        a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
        plt.savefig(f'figs/crawler_heading-{self.params["Heading"]}_swivel-{self.params["Swivel"]}.png')
        plt.show()

    def _plot_error_all(self):
        log = self.state_log
        df = log["df"]
        headings = df["h"].unique()
        swivels = df["s"].unique()

        fig_lin, axs_lin = plt.subplots(len(headings), len(swivels))
        fig_ang, axs_ang = plt.subplots(len(headings), len(swivels))
        for i, h in enumerate(headings):
            for j, s in enumerate(swivels):
                self.params["Heading"] = h
                self.params["Swivel"] = s
                sub_df = df[(df["h"]==h) & (df["s"]==s)]
                mlin = sub_df["mlin"].to_numpy()
                mang= sub_df["mang"].to_numpy()
                tlin = sub_df["tlin"].to_numpy()
                tang= sub_df["tang"].to_numpy()
                stdlin = sub_df["stdlin"].to_numpy()
                stdang = sub_df["stdang"].to_numpy()
                # nb_rows = 2
                # nb_cols = 2
                # fig_lin, axs_lin = plt.subplots(nb_rows, nb_cols)
                # fig_ang, axs_ang = plt.subplots(nb_rows, nb_cols)
                log= self.state_log
                size = log["size"]
                ticks = np.arange(size)

                fig_lin.suptitle(", ".join([f"{k}: {v}" for k, v in self.params.items()]), fontsize=8)
                a = axs_lin[0,0]
                mat = np.abs(mlin - tlin).reshape(-1, size)
                img = a.imshow(mat, cmap="jet")
                a.set_xticks(ticks)
                a.set_yticks(ticks)
                a.set_xticklabels(np.round(np.unique(tlin), 4), rotation=45)
                a.set_yticklabels(np.round(np.unique(tang), 4))
                a.tick_params(axis='both', which='major', labelsize=6)
                plt.colorbar(img, ax=a)
                a.set_title("Linear Velocity Error")
                a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
                
                a = axs[1,0]
                mat = np.abs(mang - tang).reshape(-1, size)
                img = a.imshow(mat, cmap="jet")
                a.set_xticks(ticks)
                a.set_yticks(ticks)
                a.set_xticklabels(np.round(np.unique(tlin), 4), rotation=45)
                a.set_yticklabels(np.round(np.unique(tang), 4))
                a.tick_params(axis='both', which='major', labelsize=6)
                plt.colorbar(img, ax=a)
                a.set_title("Angular Velocity Error")
                a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
                a = axs[0,1]
                      
                img = a.imshow(stdlin.reshape(-1, size), cmap="jet")
                a.set_xticks(ticks)
                a.set_yticks(ticks)
                a.set_xticklabels(np.round(np.unique(tlin), 4), rotation=45)
                a.set_yticklabels(np.round(np.unique(tang), 4))
                a.tick_params(axis='both', which='major', labelsize=6)
                plt.colorbar(img, ax=a)
                a.set_title("Linear Velocity StDev")
                a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
                a = axs[1,1]

                img = a.imshow(stdang.reshape(-1, size), cmap="jet")
                a.set_xticks(ticks)
                a.set_yticks(ticks)
                a.set_xticklabels(np.round(np.unique(tlin), 4), rotation=45)
                a.set_yticklabels(np.round(np.unique(tang), 4))
                a.tick_params(axis='both', which='major', labelsize=6)
                plt.colorbar(img, ax=a)
                a.set_title("Angular Velocity StDev")
                a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
                f_name = f'crawler_heading-{h}_swivel-{s}'
                plt.savefig(LOGGER_PATH + 'figs/'+ f_name + '.png')
                sub_df.to_csv(LOGGER_PATH + 'csv/' + f_name + '.csv')
                # plt.show()
                fig.clf()
                plt.close()

    def _plot_error_ind(self):
        log = self.state_log
        df = log["df"]
        headings = df["h"].unique()
        swivels = df["s"].unique()

        fig_lin, axs_lin = plt.subplots(len(headings), len(swivels))
        fig_ang, axs_ang = plt.subplots(len(headings), len(swivels))
        fig_lin.suptitle("Linear Velocity Error")
        fig_ang.suptitle("Angular Velocity Error")

        pad = 5
        for ax_lin, ax_ang, s in zip(axs_lin[0], axs_ang[0], swivels):
            ax_lin.annotate(s, xy=(0.5, 1), xytext=(0, pad), xycoords=ax_lin.yaxis.label, textcoords='offset points', size='large', ha='center', va='baseline')
            ax_ang.annotate(s, xy=(0.5, 1), xytext=(0, pad), xycoords=ax_ang.yaxis.label, textcoords='offset points', size='large', ha='center', va='baseline')

        for ax_lin, ax_ang, h in zip(axs_lin[:, 0], axs_ang[:, 0], headings):
            ax_lin.annotate(s, xy=(0, 0.5), xytext=(-ax_lin.yaxis.labelpad - pad, 0), xycoords=ax_lin.yaxis.label, textcoords='offset points', size='large', ha='center', va='baseline')
            ax_ang.annotate(s, xy=(0, 0.5), xytext=(-ax_ang.yaxis.labelpad - pad, 0), xycoords=ax_ang.yaxis.label, textcoords='offset points', size='large', ha='center', va='baseline')
 
        for i, h in enumerate(headings):
            for j, s in enumerate(swivels):
                # self.params["Heading"] = h
                # self.params["Swivel"] = s
                sub_df = df[(df["h"]==h) & (df["s"]==s)]
                mlin = sub_df["mlin"].to_numpy()
                mang= sub_df["mang"].to_numpy()
                tlin = sub_df["tlin"].to_numpy()
                tang= sub_df["tang"].to_numpy()
                stdlin = sub_df["stdlin"].to_numpy()
                stdang = sub_df["stdang"].to_numpy()
                log = self.state_log
                size = log["size"]
                ticks = np.arange(size)

                # fig_lin.suptitle(", ".join([f"{k}: {v}" for k, v in self.params.items()]), fontsize=8)
                plt.figure(1)
                a = axs_lin[i,j]
                mat = np.abs(mlin - tlin).reshape(-1, size)
                img = a.imshow(mat, cmap="jet")
                a.set_xticks(ticks)
                a.set_yticks(ticks)
                a.set_xticklabels(np.round(np.unique(tlin), 4), rotation=45)
                a.set_yticklabels(np.round(np.unique(tang), 4))
                a.tick_params(axis='both', which='major', labelsize=6)
                a.colorbar(img, ax=a)
                # a.set_title("Linear Velocity Error")
                a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
                
                a = axs_ang[i,j]
                mat = np.abs(mang - tang).reshape(-1, size)
                img = a.imshow(mat, cmap="jet")
                a.set_xticks(ticks)
                a.set_yticks(ticks)
                a.set_xticklabels(np.round(np.unique(tlin), 4), rotation=45)
                a.set_yticklabels(np.round(np.unique(tang), 4))
                a.tick_params(axis='both', which='major', labelsize=6)
                a.colorbar(img, ax=a)
                # a.set_title("Angular Velocity Error")
                a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
                # a = axs[0,1]
                #       
                # img = a.imshow(stdlin.reshape(-1, size), cmap="jet")
                # a.set_xticks(ticks)
                # a.set_yticks(ticks)
                # a.set_xticklabels(np.round(np.unique(tlin), 4), rotation=45)
                # a.set_yticklabels(np.round(np.unique(tang), 4))
                # a.tick_params(axis='both', which='major', labelsize=6)
                # plt.colorbar(img, ax=a)
                # a.set_title("Linear Velocity StDev")
                # a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
                # a = axs[1,1]
                #
                # img = a.imshow(stdang.reshape(-1, size), cmap="jet")
                # a.set_xticks(ticks)
                # a.set_yticks(ticks)
                # a.set_xticklabels(np.round(np.unique(tlin), 4), rotation=45)
                # a.set_yticklabels(np.round(np.unique(tang), 4))
                # a.tick_params(axis='both', which='major', labelsize=6)
                # plt.colorbar(img, ax=a)
                # a.set_title("Angular Velocity StDev")
                # a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')

                # f_name = f'crawler_heading-{h}_swivel-{s}'
                # plt.savefig(LOGGER_PATH + 'figs/'+ f_name + '.png')
                # sub_df.to_csv(LOGGER_PATH + 'csv/' + f_name + '.csv')
                # # plt.show()
                # fig.clf()
                # plt.close()
        
        fig_lin.savefig(LOGGER_PATH + 'figs/lin_error.png')
        sub_df.to_csv(LOGGER_PATH + 'csv/lin_error.csv')
        fig_ang.savefig(LOGGER_PATH + 'figs/ang_erro.png')
        sub_df.to_csv(LOGGER_PATH + 'csv/ang_error.csv')
        plt.show()
        # fig.clf()
        plt.close()

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()


