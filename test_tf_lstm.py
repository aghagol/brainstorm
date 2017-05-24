import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
map_fn = tf.map_fn
import matplotlib.pyplot as plt

seq_length = 10
batch_size = 5
CURVE_ORDER = 3

x = np.empty((seq_length, batch_size, 2))
for batch_number in range(batch_size):
  xloc = np.arange(seq_length).reshape(seq_length,1)-(seq_length/2)
  yloc = np.hstack([xloc**p for p in range(CURVE_ORDER+1)]).dot(np.random.randn(CURVE_ORDER+1,1))
  x[:,batch_number,0] = xloc.squeeze()
  x[:,batch_number,1] = yloc.squeeze()
y = np.empty((seq_length, batch_size, 2))
y = np.roll(x,-1,axis=0) #future locations

for batch_number in range(batch_size):
  plt.clf()
  plt.plot(x[:-1,batch_number,0],x[:-1,batch_number,1],'k.')
  plt.plot(y[:-1,batch_number,0],y[:-1,batch_number,1]-5,'r.')
  plt.pause(1)