"""
Module containing data visualisations generated from the project
"""
#%%

import pickle
import matplotlib.pyplot as plt

with open("NEAT-250.pickle", "rb") as f:
    neat = pickle.load(f)

with open("HYPERNEAT-250.pickle", "rb") as f:
    es_hyperneat = pickle.load(f)
    
y = list(range(len(neat["std_dev"])))

plt.plot(y, neat["best_each_gen"])
plt.plot(y, es_hyperneat["best_each_gen"])
