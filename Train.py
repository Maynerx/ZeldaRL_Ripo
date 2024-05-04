from Rom.Zelda_env import *
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from Rom.SaveOnBestCallback import SaveOnBestTrainingRewardCallback
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
        env = ZeldaEnv(rank, save=False)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init



if __name__ == '__main__':
    
    #ep_length = 2048*2
    timesteps = int(1.2e6)
    #learn_steps = 5
    num_cpu = 10
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    pre_trained = False

    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])


    vec_env = VecFrameStack(vec_env, n_stack=4)

    vec_env = VecMonitor(vec_env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=4096, log_dir=log_dir)

    
    if pre_trained:
        model = PPO.load('best_model', env=vec_env)
        model.set_parameters('best_model')
    else:
        model = PPO('CnnPolicy', env=vec_env,  n_steps=2048, batch_size=512, n_epochs=1, gamma=0.999)
        model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
        model.save('end_model')

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "ZeldaTest")
    plt.show()
    
    """
    for _ in range(learn_steps):
        model.learn(total_timesteps=ep_length*num_cpu, progress_bar=False, callback=callback)
        print(_)

    
    
    """
