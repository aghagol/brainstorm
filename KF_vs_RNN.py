import os,sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from filterpy.kalman import KalmanFilter
import tensorflow as tf
###################################### data parameters
CURVE_ORDER = 3
CURVE_STEP = .5
SEQ_LENGTH = 30
POINT_DIM = 5
OBS_DEPTH = 1
BATCH_SIZE = 100
###################################### load the TF session
MODEL_SUFFIX = ''
MODEL_SUFFIX = '.tmp'
print('Loading a saved TF session')
saver = tf.train.import_meta_graph('rnntracker-polyn%d%s.meta'%(CURVE_ORDER,MODEL_SUFFIX))
session = tf.Session()
saver.restore(session,'rnntracker-polyn%d%s'%(CURVE_ORDER,MODEL_SUFFIX))
print('...done')
###################################### helper functions
def gen_data_batch_backup(seq_length,batch_size,curve_order,curve_step,hist_depth):
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
def gen_data_batch(seq_length,batch_size,curve_order,curve_step,obs_depth,point_dim):
  x = np.zeros((seq_length,batch_size,point_dim*obs_depth))
  for sample_number in range(batch_size):
    xloc = np.sort(np.random.rand(seq_length)-.5).reshape(seq_length,1)*seq_length*curve_step
    yloc = np.hstack([xloc**p for p in range(curve_order+1)]).dot(np.random.randn(curve_order+1,1))
    x[:,sample_number,0] = xloc.squeeze()
    x[:,sample_number,1] = yloc.squeeze()
  dt_next = np.sqrt(((np.roll(x[:,:,:2],-1,axis=0)-x[:,:,:2])**2).sum(axis=2))
  x[:,:,point_dim-1] = dt_next
  if point_dim==5:
    dt_prev = np.sqrt(((np.roll(x[:,:,:2],+1,axis=0)-x[:,:,:2])**2).sum(axis=2))
    x[:,:,2] = (np.roll(x[:,:,0],+1,axis=0)-x[:,:,0]) /dt_prev *dt_next #linear (naive) estimation using previous observation
    x[:,:,3] = (np.roll(x[:,:,1],+1,axis=0)-x[:,:,1]) /dt_prev *dt_next #linear (naive) estimation using previous observation
  for step_back in range(1,obs_depth):
    x[:,:,point_dim*step_back:point_dim*(step_back+1)] = np.roll(x[:,:,:point_dim],-step_back,axis=0)
  y = np.roll(x[:,:,:2],-obs_depth,axis=0) #future locations
  return x[1:-obs_depth,:,:], y[1:-obs_depth,:,:]
def kf_tracker(kf_instance,seq):
  kf_instance.H = np.array([[1,0,0,0],[0,1,0,0]])
  kf_instance.Q *= 1
  kf_instance.R *= 0
  kf_instance.x[:2] = seq[0,:2].reshape(2,1) #initialization
  kf_instance.x[2:] = ((seq[1,:2]-seq[0,:2])/seq[0,2]).reshape(2,1) #cheating!
  kf_instance.F = np.array([[1,0,seq[0,2],0],[0,1,0,seq[0,2]],[0,0,1,0],[0,0,0,1]])
  kf_instance.predict()
  kf_instance.update(seq[1,:2].reshape(2,1))
  statelog = []
  for i in range(2,seq.shape[0]):
    kf_instance.F = np.array([[1,0,seq[i-1,2],0],[0,1,0,seq[i-1,2]],[0,0,1,0],[0,0,0,1]])
    kf_instance.predict()
    statelog.append(kf_instance.x.T)
    kf_instance.update(seq[i,:2].reshape(2,1))
  # kf_instance.F = np.array([[1,0,seq[-1,2],0],[0,1,0,seq[-1,2]],[0,0,1,0],[0,0,0,1]])
  # kf_instance.predict()
  # statelog.append(kf_instance.x.T)
  return np.vstack(statelog)
###################################### generate data
valid_x, valid_y = gen_data_batch(SEQ_LENGTH,BATCH_SIZE,CURVE_ORDER,CURVE_STEP,OBS_DEPTH,POINT_DIM)
valid_y_rnn = session.run("prediction:0",{"inputs:0":valid_x,"out_keep_prob:0":1})
fig,ax = plt.subplots(1,3,figsize=(17,9))
color = np.random.rand(100,3)
for sample_idx in range(BATCH_SIZE):
  for axi in ax: axi.cla()
  for axi in ax[0:]:
    axi.plot(valid_y[:1-max(OBS_DEPTH,2),sample_idx,0],valid_y[:1-max(OBS_DEPTH,2),sample_idx,1],'k.')
    for ll in range(OBS_DEPTH): axi.plot(valid_x[0,sample_idx,POINT_DIM*ll],valid_x[0,sample_idx,POINT_DIM*ll+1],'b.')
  # for i,point in enumerate(valid_x[:,sample_idx,:].squeeze()):
  #   ax[0].plot(point[0],point[1],'.',color=color[i])
  #   ax[0].add_patch(plt.Circle((point[0],point[1]),radius=point[POINT_DIM-1],fill=False,color=color[i]))
  ###################################### track using KF
  statelog = kf_tracker(KalmanFilter(dim_x=4,dim_z=2),valid_x[:,sample_idx,[0,1,POINT_DIM-1]].reshape(valid_x.shape[0],3))
  ax[1].plot(statelog[:,0],statelog[:,1],'o',color='red',mfc='none')
  ax[1].plot(statelog[:,0],statelog[:,1],color='green')
  ax[1].set_title('KF')
  ###################################### track using RNN
  ax[2].plot(valid_y_rnn[:1-max(OBS_DEPTH,2),sample_idx,0],valid_y_rnn[:1-max(OBS_DEPTH,2),sample_idx,1],'ro',mfc='None')
  ax[2].plot(valid_y_rnn[:1-max(OBS_DEPTH,2),sample_idx,0],valid_y_rnn[:1-max(OBS_DEPTH,2),sample_idx,1],'g')
  ax[2].set_title('RNN')
  ###################################### display for a few seconds
  # ax[0].axis('image')
  plt.pause(5)