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
SEQ_LENGTH = 25
HIST_DEPTH = 5
###################################### generate data
valid_x, valid_y = gen_data_batch(SEQ_LENGTH,1,CURVE_ORDER,CURVE_STEP,HIST_DEPTH)
###################################### track using KF
from filterpy.kalman import KalmanFilter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.H = np.array([[1,0,0,0],[0,1,0,0]])
kf.Q *= 1
kf.R *= 1
kf.x[0],kf.x[1] = valid_x[0,0,0],valid_x[0,0,1] #initialization
dt = valid_x[0,0,3]
kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
kf.predict()
statelog = [kf.x.T]
plt.plot(valid_x[:,0,0],valid_x[:,0,1],'.',color='black')
kf.update(valid_x[1,0,:2].reshape(2,1))
for i in range(2,valid_x.shape[0]):
  dt = valid_x[i,0,3]
  kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
  kf.predict()
  statelog.append(kf.x.T)
  plt.plot(kf.x[0],kf.x[1],'o',color='red',mfc='none')
  kf.update(valid_x[i,0,:2].reshape(2,1))
statelog = np.vstack(statelog)
plt.plot(statelog[:,0],statelog[:,1],color='green')
plt.show()
###################################### track using RNN
# import tensorflow as tf
