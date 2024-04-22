import numpy as np
import mediapy
from pyboy import PyBoy

pyboy = PyBoy('rom/bin/zelda.gbc')

image = pyboy.screen.ndarray
'''
image = np.delete(arr=image, obj=3, axis=2)
print(image.shape, image.dtype);quit()
image = np.random.randint(0, 256, size=(80, 80, 3), dtype='uint8')
'''
s = mediapy.VideoWriter('vid/', (80, 80))
s.__enter__()

s.add_image(image)