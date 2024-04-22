import matplotlib.pyplot as plt
import numpy as np
import mediapy as media
#from pathlib import Path
from tqdm import tqdm
import os
#runs_dir = Path('vid/')

vid_sections = []
def parse_reward(p):
    return float(p.name.split('_')[2])




for r in tqdm(sorted(os.listdir('vid/'))):
    vid_sections.append(media.read_video('vid/' + r))