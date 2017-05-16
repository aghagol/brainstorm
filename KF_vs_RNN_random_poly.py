import os,sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#generate sequence
x = np.arange(-2,2,.1).reshape(-1,1)
y = np.hstack((np.ones_like(x),x,x**2,x**3)).dot(np.random.rand(4,1))

#track using KF
from filterpy.kalman import KalmanFilter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
kf.H = np.array([[1,0,0,0],[0,1,0,0]])
kf.Q *= .001
kf.R *= 100
kf.x[0],kf.x[1] = x[1],y[1] #initialization
kf.x[2],kf.x[3] = x[1]-x[0],y[1]-y[0]
plt.plot(x,y,'.',color='k');plt.pause(.1)
plt.plot(kf.x[0],kf.x[1],'s',color='red',mfc='none');plt.pause(.1)
for i in range(2,x.shape[0]):
  kf.predict()
  if not i%3: kf.update(np.array([x[i],y[i]]).reshape(2,1))
  plt.plot(kf.x[0],kf.x[1],'s',color='red',mfc='none');plt.pause(.1)
plt.pause(.5)

#track using RNN
