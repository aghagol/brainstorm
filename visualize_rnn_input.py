import os,sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
###################################### helper functions
def gen_data_batch(seq_length,batch_size,curve_order,curve_step,hist_depth):
  x = np.empty((seq_length,batch_size,3*hist_depth))
  for sample_number in range(batch_size):
    xloc = np.sort(np.random.rand(seq_length)-.5).reshape(seq_length,1)*seq_length/2*curve_step
    yloc = np.hstack([xloc**p for p in range(curve_order+1)]).dot(np.random.randn(curve_order+1,1))
    x[:,sample_number,0] = xloc.squeeze()
    x[:,sample_number,1] = yloc.squeeze()
  x[:,:,2] = np.sqrt(((np.roll(x[:,:,:2],-1,axis=0)-x[:,:,:2])**2).sum(axis=2))
  for level_down in range(1,hist_depth):
    x[:,:,3*level_down:3*(level_down+1)] = np.roll(x[:,:,:3],-1*level_down,axis=0)
  y = np.roll(x[:,:,:2],-hist_depth,axis=0) #future locations
  return x[:-1*hist_depth,:,:], y[:-1*hist_depth,:,:]
###################################### data parameters
CURVE_ORDER = 2
CURVE_STEP = .5
SEQ_LENGTH = 20
HIST_DEPTH = 2
BATCH_SIZE = 1
###################################### generate data
valid_x, valid_y = gen_data_batch(SEQ_LENGTH,BATCH_SIZE,CURVE_ORDER,CURVE_STEP,HIST_DEPTH)
fig,ax = plt.subplots(1,1,figsize=(7,7))
color = np.random.rand(100,3)
for i,point in enumerate(valid_x[:,0,:3].squeeze()):
  ax.plot(point[0],point[1],'.',color=color[i])
  ax.add_patch(plt.Circle((point[0],point[1]),radius=point[2],fill=False,color=color[i]))
ax.axis('image')
plt.show()