from Zelda_env import *
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from SaveOnBestCallback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
import os
import matplotlib.pyplot as plt



def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ZeldaEnv(rank)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init



if __name__ == '__main__':
    
    timesteps = int(6e5)
    num_cpu = 10
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])


    vec_env = VecFrameStack(vec_env, n_stack=4)

    vec_env = VecMonitor(vec_env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=3000, log_dir=log_dir)

    model = PPO('CnnPolicy', env=vec_env) #PPO.load('best_model', env=vec_env)#
    
    model.learn(timesteps, progress_bar=True, callback=callback)

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "ZeldaTest")
    plt.show()


