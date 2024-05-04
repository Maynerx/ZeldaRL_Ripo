from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
import os
import matplotlib.pyplot as plt


ep_length = 2048*8
learn_steps = 5
num_cpu = 10
log_dir = "tmp/"


plot_results([log_dir], int(5e5), results_plotter.X_TIMESTEPS, "ZeldaTest")
plt.show()