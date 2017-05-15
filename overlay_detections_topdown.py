import os, sys
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fuse_path = '/home/mo/data_lifeng/sampled_training_data_5-4/Lane/fuse_with_pixel_coord/'
data_path = '/media/mo/remoteTitan/home/deeplearning/dataset_on_titan-pascal/topdown_urban/4-27/originalImages/'

image_list = [i[:-5] for i in os.listdir(fuse_path) if i.endswith('.fuse')]

fig, ax = plt.subplots(1,1,figsize=(10,10))
for image_name in image_list:
  try:
    ax.cla()
    with open(fuse_path+image_name+'.fuse') as ffuse:
      points = np.loadtxt(ffuse)
    image = mpimg.imread(data_path+image_name)
    ax.imshow(image)
    ax.plot(points[:,0],points[:,1],'.',color='red')
    plt.pause(2)
  except KeyboardInterrupt:
    print ""
    break

