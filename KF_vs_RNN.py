import os,sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

state_order = 1
curve_order = 3
update_step = 10

#generate sequence
x = np.arange(-10,20,1).reshape(-1,1)
y = np.hstack([x**p for p in range(curve_order+1)]).dot(np.random.randn(curve_order+1,1))

#track using KF
from filterpy.kalman import KalmanFilter
if state_order==2:
  kf = KalmanFilter(dim_x=6, dim_z=2)
  kf.F = np.array([[1,0,1,0,.5,0],[0,1,0,1,0,.5],[0,0,1,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0],[0,0,0,0,0,1]])
  kf.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])
else:
  kf = KalmanFilter(dim_x=4, dim_z=2)
  kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
  kf.H = np.array([[1,0,0,0],[0,1,0,0]])
kf.Q *= 1
kf.R *= 1
kf.x[0],kf.x[1] = x[0],y[0] #initialization
plt.plot(x,y,'.',color='black');plt.pause(.1)
plt.plot(kf.x[0],kf.x[1],'s',color='red',mfc='none');plt.pause(.1)
statelog = [kf.x.T]
for i in range(1,x.shape[0]):
  kf.predict()
  if i<3 or not i%update_step:
    kf.update(np.array([x[i],y[i]]).reshape(2,1))
    plt.plot(kf.x[0],kf.x[1],'s',color='blue',mfc='none');plt.pause(.1)
  else:
    plt.plot(kf.x[0],kf.x[1],'s',color='red',mfc='none');plt.pause(.1)
  statelog.append(kf.x.T)
statelog = np.vstack(statelog)
plt.plot(statelog[:,0],statelog[:,1],color='green');plt.pause(.1)
plt.pause(1)

# #track using RNN
# import tensorflow as tf
