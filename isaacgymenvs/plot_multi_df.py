import matplotlib.pyplot as plt
import numpy as np
from plotter import Plotter
import pandas as pd

plt.rcParams["figure.figsize"] = (12,10)
plt.rcParams["font.family"] = ["Gulasch", "Times", "Times New Roman", "serif"]
plt.rcParams["font.size"] = 12

df=[]
lstm_df = []
ivk_df = []
df.append(pd.read_csv('runs/crawler-ideal/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
df.append(pd.read_csv('runs/crawler-noisy/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
df.append(pd.read_csv('runs/crawler-constant/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
df.append(pd.read_csv('runs/crawler-tether1/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
df.append(pd.read_csv('runs/crawler-tether2/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
df.append(pd.read_csv('runs/crawler-tether3/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
lstm_df.append(pd.read_csv('runs/crawler-ideal-lstm/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
lstm_df.append(pd.read_csv('runs/crawler-noisy-lstm/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
lstm_df.append(pd.read_csv('runs/crawler-constant-lstm/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
lstm_df.append(pd.read_csv('runs/crawler-tether1-lstm/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
lstm_df.append(pd.read_csv('runs/crawler-tether2-lstm/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
lstm_df.append(pd.read_csv('runs/crawler-tether3-lstm/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
ivk_df.append(pd.read_csv('runs/crawler-ideal-ivk/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
ivk_df.append(pd.read_csv('runs/crawler-noisy-ivk/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
ivk_df.append(pd.read_csv('runs/crawler-constant-ivk/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
ivk_df.append(pd.read_csv('runs/crawler-tether1-ivk/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
ivk_df.append(pd.read_csv('runs/crawler-tether2-ivk/data/csv/crawler_heading-0.0_swivel-0.0.csv'))
ivk_df.append(pd.read_csv('runs/crawler-tether3-ivk/data/csv/crawler_heading-0.0_swivel-0.0.csv'))

tlin = df[0]["tlin"].unique()
tang = df[0]["tang"].unique()

lin_rmse = []
lin_std = []
ang_rmse = []
ang_std = []

for d in df:
    lr =  d["rmse_lin"].mean()
    lstd = d["rmse_lin"].std()
    ar = d["rmse_ang"].mean()
    astd = d["rmse_ang"].std()
    lin_rmse.append(lr) 
    ang_rmse.append(ar)
    lin_std.append(lstd)
    ang_std.append(astd)


lstm_lin_rmse = []
lstm_lin_std = []
lstm_ang_rmse = []
lstm_ang_std = []
for d in lstm_df:
    lr =  d["rmse_lin"].mean()
    lstd = d["rmse_lin"].std()
    ar = d["rmse_ang"].mean()
    astd = d["rmse_ang"].std()
    lstm_lin_rmse.append(lr) 
    lstm_ang_rmse.append(ar)
    lstm_lin_std.append(lstd)
    lstm_ang_std.append(astd)

ivk_lin_rmse = []
ivk_lin_std = []
ivk_ang_rmse = []
ivk_ang_std = []
for d in ivk_df:
    lr =  d["rmse_lin"].mean()
    lstd = d["rmse_lin"].std()
    ar = d["rmse_ang"].mean()
    astd = d["rmse_ang"].std()
    ivk_lin_rmse.append(lr) 
    ivk_ang_rmse.append(ar)
    ivk_lin_std.append(lstd)
    ivk_ang_std.append(astd)
# for d in df:
#     lr =  ((d["tlin"] - d["mlin"]) ** 2).mean() ** 0.5
#     lstd = (d["tlin"] - d["mlin"]).abs().std()
#     ar =  ((d["tang"] - d["mang"]) ** 2).mean() ** 0.5
#     astd = (d["tang"] - d["mang"]).std()
#     lin_rmse.append(lr) 
#     ang_rmse.append(ar)
#     lin_std.append(lstd)
#     ang_std.append(astd)
#
#
# lstm_lin_rmse = []
# lstm_lin_std = []
# lstm_ang_rmse = []
# lstm_ang_std = []
# for d in lstm_df:
#     lr =  ((d["tlin"] - d["mlin"]) ** 2).mean() ** 0.5
#     lstd = (d["tlin"] - d["mlin"]).std()
#     ar =  ((d["tang"] - d["mang"]) ** 2).mean() ** 0.5
#     astd = (d["tang"] - d["mang"]).std()
#     lstm_lin_rmse.append(lr) 
#     lstm_ang_rmse.append(ar)
#     lstm_lin_std.append(lstd)
#     lstm_ang_std.append(astd)
index = np.arange(6)
bar_width = 0.25

# fig, ax = plt.subplots()
# ax.bar(np.arange(len(df)), lin_rmse, bar_width, label="PPO", color="royalblue")
#
# ax.bar(np.arange(len(lstm_df))+bar_width, lstm_lin_rmse, bar_width, 
#                  label="PPO-LSTM", color="coral")
#
# ax.set_xlabel('Environment')
# ax.set_ylabel('RMSE')
# ax.set_title('Linear Velocity RMSE by Environment, Algorithm')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(["Ideal", "Noisy", "Noisy+Constant", "Noisy+Constant Tether", "Noisy+Impulse Tether", "Noisy+Sin Tether"], rotation=45)
# ax.legend()
#
# plt.savefig("barchart_lin.png", bbox_inches='tight')
# plt.show()
#
# fig, ax = plt.subplots()
# summer = ax.bar(np.arange(len(df)), ang_rmse, bar_width, label="PPO", color="mediumpurple")
#
# winter = ax.bar(np.arange(len(lstm_df))+bar_width, lstm_ang_rmse, bar_width,
#                  label="PPO-LSTM", color="darkorange")
#
# ax.set_xlabel('Environment')
# ax.set_ylabel('RMSE')
# ax.set_title('Angular Velocity RMSE by Environment, Algorithm')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(["Ideal", "Noisy", "Noisy+Constant", "Noisy+Constant Tether", "Noisy+Impulse Tether", "Noisy+Sin Tether"], rotation=45)
# ax.legend()
# plt.savefig("barchart_ang.png", bbox_inches='tight')
# plt.show()
fig, ax = plt.subplots()
ax.bar(np.arange(len(df))-bar_width, lin_rmse, bar_width, yerr=lin_std, capsize=5, label="PPO", color="royalblue")

ax.bar(np.arange(len(lstm_df)), lstm_lin_rmse, bar_width, yerr=lstm_lin_std, capsize=5,
                 label="PPO-LSTM", color="coral")

ax.bar(np.arange(len(df))+bar_width, ivk_lin_rmse, bar_width, yerr=ivk_lin_std, capsize=5, label="IVK", color="forestgreen")

ax.set_xlabel('Environment')
ax.set_ylabel('RMSE')
ax.set_title('Linear Velocity RMSE by Environment, Method')
ax.set_xticks(index)
ax.set_xticklabels(["Ideal", "Noisy", "Noisy+Constant", "Noisy+Constant Tether", "Noisy+Impulse Tether", "Noisy+Sin Tether"], rotation=45)
ax.legend()

plt.savefig("barchart_lin.png", bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.bar(np.arange(len(df))-bar_width, ang_rmse, bar_width, yerr=ang_std, capsize=5, label="PPO", color="royalblue")

ax.bar(np.arange(len(lstm_df)), lstm_ang_rmse, bar_width, yerr=lstm_ang_std, capsize=5,
                 label="PPO-LSTM", color="coral")

ax.bar(np.arange(len(df))+bar_width, ivk_ang_rmse, bar_width, yerr=ivk_ang_std, capsize=5, label="IVK", color="forestgreen")

ax.set_xlabel('Environment')
ax.set_ylabel('RMSE')
ax.set_title('Angular Velocity RMSE by Environment, Method')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(["Ideal", "Noisy", "Noisy+Constant", "Noisy+Constant Tether", "Noisy+Impulse Tether", "Noisy+Sin Tether"], rotation=45)
ax.legend()
plt.savefig("barchart_ang.png", bbox_inches='tight')
plt.show()

cross_df = pd.read_csv("runs/crawler-ideal/data-cross/csv/crawler_heading-0.0_swivel-0.0.csv")
same_df = pd.read_csv('runs/crawler-tether1/data/csv/crawler_heading-0.0_swivel-0.0.csv')

tlin = cross_df["tlin"].unique()
cross_group = cross_df.groupby("tlin")
same_group = same_df.groupby("tlin")
c_lin_mean = cross_group["mlin"].mean()
c_ang_mean = cross_group["mang"].mean()
s_lin_mean = same_group["mlin"].mean()
s_ang_mean = same_group["mang"].mean()

clin_rmse = ((cross_df["tlin"] - cross_df["mlin"]) ** 2).mean() ** 0.5 
cang_rmse = ((cross_df["tang"] - cross_df["mang"]) ** 2).mean() ** 0.5 

slin_rmse = ((same_df["tlin"] - same_df["mlin"]) ** 2).mean() ** 0.5 
sang_rmse = ((same_df["tang"] - same_df["mang"]) ** 2).mean() ** 0.5 

fig, ax = plt.subplots(1, 2)
fig.suptitle('Model Resilience in Noisy Environment w/ Constant Tether Disturbance')

a = ax[0]
index=np.arange(1)
a.bar(np.arange(1), slin_rmse, bar_width, label="same", color="royalblue")
a.bar(np.arange(1)+bar_width, clin_rmse, bar_width, label="cross", color="coral")
# a.set_xlabel('Model Type')
a.set_ylabel('Linear Velocity RMSE')
a.set_xticks([])
# a.set_title('city RMSE by Environment, Algorithm')
# a.set_xticks(index + bar_width / 2)
# a.set_xticklabels(["Ideal", "Noisy", "Noisy+Constant", "Noisy+Constant Tether", "Noisy+Impulse Tether", "Noisy+Sin Tether"], rotation=45)
a.legend(loc="upper right")

a = ax[1]
a.bar(np.arange(1), sang_rmse, bar_width, label="same", color="royalblue")
a.bar(np.arange(1)+bar_width, cang_rmse, bar_width, label="cross", color="coral")
# a.set_xlabel('Model Type')
a.set_ylabel('Angular Velocity RMSE')
# a.set_title('city RMSE by Environment, Algorithm')
a.set_xticks([])
# a.set_xticklabels(["Ideal", "Noisy", "Noisy+Constant", "Noisy+Constant Tether", "Noisy+Impulse Tether", "Noisy+Sin Tether"], rotation=45)
a.legend(loc="upper right")
# fig, ax = plt.subplots(2, 1)
# fig.suptitle('Model Resilience in Noisy Environment w/ Constant Tether Disturbance')
# a = ax[0]
# summer = a.bar(np.arange(len(tlin)), c_lin_mean, bar_width, label="Cross", color="mediumpurple")
#
# winter = a.bar(np.arange(len(tlin))+bar_width, s_lin_mean,
#                  bar_width, label="Same", color="darkorange")
#
# a.set_xlabel('Linear Velocity')
# a.set_ylabel('Linear Velocity RMSE')
# a.set_xticks(np.arange(len(tlin)) + bar_width / 2)
# a.set_xticklabels(tlin)
# a.legend()
#
# a = ax[1]
#
# summer = a.bar(np.arange(len(tang)), c_ang_mean, bar_width, label="Cross", color="mediumpurple")
#
# winter = a.bar(np.arange(len(tang))+bar_width, s_ang_mean,
#                  bar_width, label="Same", color="darkorange")
#
# a.set_xlabel('Angular Velocity')
# a.set_ylabel('Angular Velocity RMSE')
# a.set_xticks(np.arange(len(tang)) + bar_width / 2)
# a.set_xticklabels(tang)
# a.legend()
# plt.savefig("cross_barchart.png")
plt.show()
