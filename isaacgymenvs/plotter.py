import os
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
        plt.rcParams["figure.figsize"] = (10,10)
        plt.rcParams["font.family"] = ["Gulasch", "Times", "Times New Roman", "serif"]
        plt.rcParams["font.size"] = 12

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
        log = self.state_log
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

    def _plot_boxplot(self):
        save_df = False
        log = self.state_log
        df = log["df"]

        data_dir = os.path.join(log["experiment_dir"], "data/")
        figs_dir = os.path.join(data_dir, 'figs/')
        csv_dir = os.path.join(data_dir, 'csv/')
        os.makedirs(figs_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        lin_grouped = df.groupby("tlin")
        ang_grouped = df.groupby("tang")
        tlin = [round(x, 2) for x in lin_grouped.groups.keys()]
        tang = [round(x, 2) for x in ang_grouped.groups.keys()]
        lin_data = [lin_grouped.get_group(key)["mlin"] for key in lin_grouped.groups.keys()]
        ang_data = [ang_grouped.get_group(key)["mang"] for key in ang_grouped.groups.keys()]

        nb_rows = 2
        nb_cols = 1
        fig, axs = plt.subplots(nb_rows, nb_cols)
        # for key, value in self.state_log.items():
        #     time = np.linspace(0, len(value)*self.dt, len(value))
        #     break
        log= self.state_log
        
        fig.suptitle("Noisy Environment w/ Constant Tether, Cross-Evaluation", fontsize=14)
        a = axs[0]
        a.boxplot(lin_data, positions=tlin, widths=0.01)
        # a.set_xticks(range(len(tlin)))
        # a.set_xticklabels(tlin)
        a.plot(tlin, tlin, color="green", label="target")
        a.set_xlim(-0.22, 0.22)
        a.tick_params(axis='x', labelrotation = 45)
        a.set(xlabel='Target Linear Velocity', ylabel='Measured Linear Velocity')
        a.legend()
        a = axs[1]
        a.boxplot(ang_data, positions=tang, widths=0.075)
        # a.set_xticks(range(len(tang)))
        # a.set_xticklabels(tang)
        a.plot(tang, tang, color="green", label="target")
        a.set_xlim(-1.1, 1.1)
        a.tick_params(axis='x', labelrotation = 45)
        a.set(xlabel='Target Angular Velocity', ylabel='Measured Angular Velocity')
        a.legend()
        plt.savefig(figs_dir + 'cross-boxplot.png')
        plt.show()

    def _plot_errorbar(self):
        save_df = False
        log = self.state_log
        df = log["df"]

        data_dir = os.path.join(log["experiment_dir"], "data/")
        figs_dir = os.path.join(data_dir, 'figs/')
        csv_dir = os.path.join(data_dir, 'csv/')
        os.makedirs(figs_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        # h, s = (0.0, 0.0)
        # sub_df = df[(df["h"]==h) & (df["s"]==s)]
        lin_grouped = df.groupby("tlin")
        ang_grouped = df.groupby("tang")
        tlin = [round(x, 2) for x in lin_grouped.groups.keys()]
        tang = [round(x, 2) for x in ang_grouped.groups.keys()]
        mlin_mean = lin_grouped["mlin"].mean()
        mang_mean = ang_grouped["mang"].mean()
        stdlin_mean = lin_grouped["stdlin"].mean()
        stdang_mean = ang_grouped["stdang"].mean()

        # mlin = sub_df["mlin"].to_numpy()
        # mang= sub_df["mang"].to_numpy()
        # tlin = sub_df["tlin"].to_numpy()
        # tang= sub_df["tang"].to_numpy()
        # stdlin = sub_df["stdlin"].to_numpy()
        # stdang = sub_df["stdang"].to_numpy()

        nb_rows = 2
        nb_cols = 1
        fig, axs = plt.subplots(nb_rows, nb_cols)
        # for key, value in self.state_log.items():
        #     time = np.linspace(0, len(value)*self.dt, len(value))
        #     break
        log= self.state_log
        
        # fig.suptitle(f"{self.params['title']}", fontsize=14)
        a = axs[0]
        a.errorbar(tlin, mlin_mean, yerr=stdlin_mean, fmt="o", color="teal", label="measured")
        a.plot(tlin, tlin, color="orange", label="target")
        a.set(xlabel='Target Linear Velocity', ylabel='Measured Linear Velocity')
        a.legend()
        a = axs[1]
        a.errorbar(tang, mang_mean, yerr=stdang_mean, fmt="o", color="firebrick", label="measured")
        a.plot(tang, tang, color="orange", label="target")
        a.set(xlabel='Target Angular Velocity', ylabel='Measured Angular Velocity')
        a.legend()
        plt.savefig(figs_dir + 'errorbar.png')
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
                sub_df.to_csv(LOGGER_PATH + 'csv/' + f_name + '.csv', mode='w+')
                # plt.show()
                fig.clf()
                plt.close()

    def _plot_error_ind(self):
        save_df = True 
        log = self.state_log
        df = log["df"]

        data_dir = os.path.join(log["experiment_dir"], "data/")
        figs_dir = os.path.join(data_dir, 'figs/')
        csv_dir = os.path.join(data_dir, 'csv/')
        os.makedirs(figs_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        headings = df["h"].unique()
        swivels = df["s"].unique()

        fig_lin, axs_lin = plt.subplots(len(headings), len(swivels))
        fig_ang, axs_ang = plt.subplots(len(headings), len(swivels))
        fig_lin.suptitle("Linear Velocity Error")
        fig_ang.suptitle("Angular Velocity Error")

        pad = 5
        for ax_lin, ax_ang, s in zip(axs_lin[0], axs_ang[0], swivels):
            ax_lin.annotate(round(s, 2), xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
            ax_ang.annotate(round(s, 2), xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')

        for ax_lin, ax_ang, h in zip(axs_lin[:, 0], axs_ang[:, 0], headings):
            ax_lin.annotate(round(h, 2), xy=(0, 0.5), xytext=(-ax_lin.yaxis.labelpad - pad, 0), xycoords=ax_lin.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
            ax_ang.annotate(round(h, 2), xy=(0, 0.5), xytext=(-ax_ang.yaxis.labelpad - pad, 0), xycoords=ax_ang.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
        
        fig_lin.text(0.5, 0.96, "Swivel Position", ha='center', va='center') 
        fig_ang.text(0.5, 0.96, "Swivel Position", ha='center', va='center') 
        fig_lin.text(0.02, 0.5, "Heading", va='center', ha='center', rotation='vertical')
        fig_ang.text(0.02, 0.5, "Heading", va='center', ha='center', rotation='vertical')
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
                mat = mat[::-1, :] 
                img = a.imshow(mat, cmap="jet", vmin=0.0, vmax=0.2)
                # a.set_xticks(ticks)
                # a.set_yticks(ticks)
                # a.set_xticklabels(np.round(np.unique(tlin), 4), rotation=45)
                # a.set_yticklabels(np.round(np.unique(tang), 4))
                a.set_xticks(ticks)
                a.set_yticks(ticks)
                a.set_xticklabels(np.round(np.unique(tang), 4), rotation=45)
                a.set_yticklabels(np.round(np.unique(tlin), 4)[::-1])
                a.tick_params(axis='both', which='major', labelsize=6)
                fig_lin.colorbar(img, ax=a)
                # a.set_title("Linear Velocity Error")
                # a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
                
                a = axs_ang[i,j]
                mat = np.abs(mang - tang).reshape(-1, size)
                mat = mat[::-1, :] 
                img = a.imshow(mat, cmap="jet", vmin=0.0, vmax=1.0)
                # a.set_xticks(ticks)
                # a.set_yticks(ticks)
                # a.set_xticklabels(np.round(np.unique(tlin), 4), rotation=45)
                # a.set_yticklabels(np.round(np.unique(tang), 4))
                a.set_xticks(ticks)
                a.set_yticks(ticks)
                a.set_xticklabels(np.round(np.unique(tang), 4), rotation=45)
                a.set_yticklabels(np.round(np.unique(tlin), 4)[::-1])
                a.tick_params(axis='both', which='major', labelsize=6)
                fig_ang.colorbar(img, ax=a)
                # a.set_title("Angular Velocity Error")
                # a.set(xlabel='Target Linear Velocity', ylabel='Target Angular Velocity')
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

                f_name = f'crawler_heading-{h}_swivel-{s}'
                # plt.savefig(LOGGER_PATH + 'figs/'+ f_name + '.png')
                if save_df:
                    sub_df.to_csv(csv_dir + f_name + '.csv', mode='w+')
                # # plt.show()
                # fig.clf()
                # plt.close()
        
        fig_lin.tight_layout()
        fig_ang.tight_layout()
        fig_lin.subplots_adjust(left=0.05, top=0.95)
        fig_ang.subplots_adjust(left=0.05, top=0.95)
        if save_df:
            df.to_csv(csv_dir + 'crawler_data.csv', mode='w+')
        fig_lin.savefig(figs_dir + 'lin_error.png')
        fig_ang.savefig(figs_dir + 'ang_error.png')
        fig_lin.show()
        # fig_ang.show()
        # fig.clf()
        #fig_lin.close()
        #fig_ang.show()

    def plot_vels(self): 
        log = self.state_log
        size = log["size"]
        plt.rcParams["figure.figsize"] = (10,10)
        # plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.family"] = ["Gulasch", "Times", "Times New Roman", "serif"]

        nb_rows = 2
        nb_cols = 1
        fig, axs = plt.subplots(nb_rows, nb_cols)

        data_dir = os.path.join(log["experiment_dir"], "data/")
        figs_dir = os.path.join(data_dir, 'figs/')

        ticks = np.arange(size) 
        
        # fig.suptitle# ("Velocity Control in a Noisy Environment")
        a = axs[0]
        x = ticks
        y0 = log["mlin"][0]
        y1 = log["mlin"][1]
        y2 = log["mlin"][2]
        y3 = log["mlin"][3]
        # y5 = log["mlin"][4]
        y4 = log["tlin"]
        a.plot(x, y0, label="$0^{\circ}$")
        a.plot(x, y1, label="$90^{\circ}$")
        a.plot(x, y2, label="$180^{\circ}$")
        a.plot(x, y3, label="$270^{\circ}$")
        # a.plot(x, y0, label="orientation 1")
        # a.plot(x, y1, label="orientation 2")
        # a.plot(x, y2, label="orientation 3")
        # a.plot(x, y3, label="orientation 4")
        # a.plot(x, y0, label="RL")
        # a.plot(x, y5, label="IVK")
        a.plot(x, y4, label="target", color="c")
        # a.set_ylim([-0.2, 0.2])
        a.legend(loc="lower right")
        # a.set_title("Linear Velocity Error")
        a.set(xlabel='Timesteps', ylabel='Linear Velocity (m/s)')
        
        a = axs[1]
        x = ticks
        y0 = log["mang"][0]
        y1 = log["mang"][1]
        y2 = log["mang"][2]
        y3 = log["mang"][3]
        y5 = log["mang"][4]
        y4 = log["tang"]
        a.plot(x, y0, label="$0^{\circ}$")
        a.plot(x, y1, label="$90^{\circ}$")
        a.plot(x, y2, label="$180^{\circ}$")
        a.plot(x, y3, label="$270^{\circ}$")
        # a.plot(x, y0, label="swivel pos. 1")
        # a.plot(x, y1, label="swivel pos. 2")
        # a.plot(x, y2, label="swivel pos. 3")
        # a.plot(x, y3, label="swivel pos. 4")
        # a.plot(x, y0, label="orientation 1")
        # a.plot(x, y1, label="orientation 2")
        # a.plot(x, y2, label="orientation 3")
        # a.plot(x, y3, label="orientation 4")
        # a.plot(x, y0, label="RL")
        # a.plot(x, y5, label="IVK")
        a.plot(x, y4, label="target", color="c")
        a.set_ylim([-0.5, 0.5])
        a.legend(loc="lower right")
        # a.set_title("Angular Velocity Error")
        a.set(xlabel='Timesteps', ylabel='Angular Velocity (rad/s)')
        
        f = figs_dir + "vels-swivel.png"
        plt.savefig(f)
        print(f"Saved figure to : {f}")
        # plt.show()

    def plot_traj(self):
        log = self.state_log
        size = log["size"]
        plt.rcParams["figure.figsize"] = (10,10)
        # plt.rcParams["font.family"] = "serif"

        nb_rows = 1
        nb_cols = 1
        fig, axs = plt.subplots(nb_rows, nb_cols)

        data_dir = os.path.join(log["experiment_dir"], "data/")
        figs_dir = os.path.join(data_dir, 'figs/')

        ticks = np.arange(size) 
        
        # fig.suptitle("Trajectory")
        a = axs
        x = log["mpos"][:,:,0]
        y = log["mpos"][:,:,1]
        # a.plot(x, y0, label="swivel pos. 1")
        # a.plot(x, y1, label="swivel pos. 2")
        # a.plot(x, y2, label="swivel pos. 3")
        # a.plot(x, y3, label="swivel pos. 4")
        # a.plot(x, y0, label="orientation 1")
        # a.plot(x, y1, label="orientation 2")
        # a.plot(x, y2, label="orientation 3")
        # a.plot(x, y3, label="orientation 4")
        # a.plot(x[0], y[0], label="$0^\circ$")
        # a.plot(y[1], -x[1], label="$90^\circ$")
        # a.plot(-x[2], -y[2], label="$180^\circ$")
        # a.plot(-y[3], -x[3], label="$270^\circ$")
        # a.plot(np.linspace(0, 2.4, len(x[0])), np.zeros(len(y[0])), label="target")
        a.plot(x[0], y[0], label="$0^\circ$")
        a.plot(x[1], y[1], label="$90^\circ$")
        a.plot(x[2], y[2], label="$180^\circ$")
        a.plot(x[3], y[3], label="$270^\circ$")
        a.plot(np.linspace(0, 2.4, len(x[0])), np.zeros(len(y[0])), label="target", color="c")
        # a.plot(x[4], y[4], label="$0^\circ$")
        # a.set_ylim([-0.2, 0.2])
        a.legend(loc="upper right")
        # a.set_title("Linear Velocity Error")
        a.set(xlabel='x', ylabel='y')
        
        f = figs_dir + "traj-swiv.png"
        plt.savefig(f)
        print(f"Saved figure to : {f}")
        # plt.show()


    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()


