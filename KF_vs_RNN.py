import os,sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# #generate sequence
# n = 50
# points = np.random.randn(n,2)
# p = np.polyfit(points[:,0],points[:,1],deg=2)
# x = np.arange(points[:,0].min(),points[:,0].max(),.1)
# y = p[0]+p[1]*x+p[2]*(x**2)

# #track using KF
# from filterpy.kalman import KalmanFilter
# kf = KalmanFilter(dim_x=4, dim_z=2)
# kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
# kf.H = np.array([[1,0,0,0],[0,1,0,0]])
# kf.Q *= .0001
# kf.R *= 100
# kf.x[0],kf.x[1] = x[1],y[1] #initialization
# kf.x[2],kf.x[3] = x[1]-x[0],y[1]-y[0]
# plt.plot(x,y,'.',color='k');plt.pause(.1)
# plt.plot(kf.x[0],kf.x[1],'s',color='red',mfc='none');plt.pause(.1)
# for i in range(2,x.shape[0]):
#   kf.predict()
#   if not i%3: kf.update(np.array([x[i],y[i]]).reshape(2,1))
#   plt.plot(kf.x[0],kf.x[1],'s',color='red',mfc='none');plt.pause(.1)
# plt.pause(1)

#track using RNN
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
