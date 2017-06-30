import numpy as np
from scipy import misc
from scipy.ndimage.filters import maximum_filter
import matplotlib.pyplot as plt

pred_im = misc.imread('1163250284379919_1163250306106648_pred.png')/(2.**16-1)
# pred_im = pred_im[1500:2000,500:1000]
peaks = (pred_im==maximum_filter(pred_im,size=20))

fig,ax = plt.subplots(1,1,figsize=(9,9))
ax.imshow(pred_im,cmap='gray')
for i in range(peaks.shape[0]):
  for j in range(peaks.shape[1]):
    if peaks[i][j] and pred_im[i][j]>.4:
      ax.plot(j,i,'g.')
plt.show()