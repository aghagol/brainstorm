import os,sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from filterpy.kalman import KalmanFilter
import tensorflow as tf
###################################### data parameters
CURVE_ORDER = 2
CURVE_STEP = .5
SEQ_LENGTH = 30
HIST_DEPTH = 2
BATCH_SIZE = 100
###################################### overhead work
print('Loading a saved TF session')
saver = tf.train.import_meta_graph('rnntracker-polyn%d.meta'%(CURVE_ORDER))
session = tf.Session()
saver.restore(session,'rnntracker-polyn%d'%(CURVE_ORDER))
print('...done')
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
  kf_instance.F = np.array([[1,0,seq[-1,2],0],[0,1,0,seq[-1,2]],[0,0,1,0],[0,0,0,1]])
  kf_instance.predict()
  statelog.append(kf_instance.x.T)
  return np.vstack(statelog)
###################################### generate data
valid_x, valid_y = gen_data_batch(SEQ_LENGTH,BATCH_SIZE,CURVE_ORDER,CURVE_STEP,HIST_DEPTH)
valid_y_rnn = session.run("prediction:0",{"inputs:0":valid_x})
fig,ax = plt.subplots(1,3,figsize=(17,9))
color = np.random.rand(100,3)
for sample_idx in range(BATCH_SIZE):
  for axi in ax: axi.cla()
  for axi in ax[1:]:
    axi.plot(valid_y[:-HIST_DEPTH+1,sample_idx,0],valid_y[:-HIST_DEPTH+1,sample_idx,1],'.',color='black')
    for ll in range(HIST_DEPTH): axi.plot(valid_x[0,sample_idx,3*ll],valid_x[0,sample_idx,3*ll+1],'.',color='blue')
  for i,point in enumerate(valid_x[:,sample_idx,:3].squeeze()):
    ax[0].plot(point[0],point[1],'.',color=color[i])
    ax[0].add_patch(plt.Circle((point[0],point[1]),radius=point[2],fill=False,color=color[i]))
  ###################################### track using KF
  statelog = kf_tracker(KalmanFilter(dim_x=4,dim_z=2),valid_x[:,sample_idx,:3].reshape(valid_x.shape[0],3))
  ax[1].plot(statelog[:,0],statelog[:,1],'o',color='red',mfc='none')
  ax[1].plot(statelog[:,0],statelog[:,1],color='green')
  ax[1].set_title('KF')
  ###################################### track using RNN
  ax[2].plot(valid_y_rnn[:-HIST_DEPTH+1,sample_idx,0],valid_y_rnn[:-HIST_DEPTH+1,sample_idx,1],'o',color='red',mfc='None')
  ax[2].plot(valid_y_rnn[:-HIST_DEPTH+1,sample_idx,0],valid_y_rnn[:-HIST_DEPTH+1,sample_idx,1],color='green')
  ax[2].set_title('RNN')
  ###################################### display for a few seconds
  ax[0].axis('image')
  plt.pause(5)