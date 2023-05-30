import numpy as np
import matplotlib.pyplot as plt

seed_min = 0
seed_max = 9

objective_table = np.load("./Data/inexact_multi_bridge_objective_record_{}.npy".format(seed_min))
regret_entry = np.amax(np.load("./Data/inexact_multi_bridge_regret_table_{}.npy".format(seed_min)), axis=1)
regret_length = regret_entry.shape[0]
regret_table = np.zeros((seed_max-seed_min+1, regret_length))
regret_table[0, :] = regret_entry
for i in range(seed_min + 1, seed_max + 1):
    regret_table[i - seed_min, :] = np.amax(np.load("./Data/inexact_multi_bridge_regret_table_{}.npy".format(i)), axis=1)
print(regret_table)

plt.figure()
for j in range(objective_table.shape[0]):
    plt.plot(objective_table[j, :], label="{}".format(j))
plt.xlabel("iteration")
plt.ylabel("reward")
plt.title("Expected utility of each agent")
plt.legend()
plt.savefig("./Figures/inexact_multi_bridge_objective.png")

show_length = regret_length

plt.figure()
x = np.linspace(0, show_length-1, show_length)
Nash_regret_table = np.zeros_like(regret_table)
for i in range(regret_length):
    Nash_regret_table[:, i:] += regret_table[:, :(regret_length-i)]
for i in range(regret_length):
    Nash_regret_table[:, i] /= (i+1)

regret_curve = np.mean(Nash_regret_table, axis=0)
std_curve = np.std(Nash_regret_table, axis=0)
plt.plot(x, regret_curve[:show_length])
plt.fill_between(x, (regret_curve-std_curve)[:show_length], (regret_curve+std_curve)[:show_length], color='cyan', alpha=0.3)
#plt.yscale("log")
plt.xlabel("iteration (*100)")
plt.ylabel("regret")
plt.title("Nash regret")
plt.savefig("./Figures/inexact_multi_bridge_max_regret.png")

regret_entry = np.mean(np.load("./Data/inexact_multi_bridge_regret_table_{}.npy".format(seed_min)), axis=1)
regret_length = regret_entry.shape[0]
regret_table = np.zeros((seed_max-seed_min+1, regret_length))
regret_table[0, :] = regret_entry
for i in range(seed_min + 1, seed_max + 1):
    regret_table[i - seed_min, :] = np.mean(np.load("./Data/inexact_multi_bridge_regret_table_{}.npy".format(i)), axis=1)

plt.figure()
x = np.linspace(0, show_length-1, show_length)

Nash_regret_table = np.zeros_like(regret_table)
for i in range(regret_length):
    Nash_regret_table[:, i:] += regret_table[:, :(regret_length-i)]
for i in range(regret_length):
    Nash_regret_table[:, i] /= (i+1)

regret_curve = np.mean(Nash_regret_table, axis=0)
std_curve = np.std(Nash_regret_table, axis=0)
plt.plot(x, regret_curve[:show_length])
plt.fill_between(x, (regret_curve-std_curve)[:show_length], (regret_curve+std_curve)[:show_length], color='cyan', alpha=0.3)
#plt.yscale("log")
plt.xlabel("iteration (*100)")
plt.ylabel("regret")
plt.title("Averaged Nash regret")
plt.savefig("./Figures/inexact_multi_bridge_mean_regret.png")

