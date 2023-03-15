"""
Module containing data visualisations generated from the project
"""
#%%

import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Opens CPPN-NEAT and HyperNEAT run files
with open("NEAT-250.pickle", "rb") as f:
    neat = pickle.load(f)

with open("HYPERNEAT-250.pickle", "rb") as f:
    es_hyperneat = pickle.load(f)

#%%
# Plots best fitness 

y = list(range(len(neat["best_each_gen"])))

plt.plot(y, neat["best_each_gen"])
plt.plot(y, es_hyperneat["best_each_gen"])
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(["CPPN-NEAT", "ES-HyperNEAT"])

#%%
# Plots std_dev

y = list(range(len(neat["std_dev"])))

plt.plot(y, neat["std_dev"])
plt.plot(y, es_hyperneat["std_dev"])

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(["CPPN-NEAT", "ES-HyperNEAT"])

# %%
# Plots mean

y = list(range(len(neat["mean"])))

plt.plot(y, neat["mean"])
plt.plot(y, es_hyperneat["mean"])

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(["CPPN-NEAT", "ES-HyperNEAT"])
# %%
