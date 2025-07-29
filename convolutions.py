import numpy as np
import scipy
import matplotlib.pyplot as plt


img = scipy.datasets.ascent().astype(np.int32)
filter = [[-1,-2,-1],[0,0,0],[1,2,1]]
weight=1

img_transformed = np.copy(img)
size_x = img_transformed.shape[0]
size_y = img_transformed.shape[1]



for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      output_pixel = 0.0
      output_pixel = output_pixel + (img[x - 1, y-1] * filter[0][0])
      output_pixel = output_pixel + (img[x, y-1] * filter[0][1])
      output_pixel = output_pixel + (img[x + 1, y-1] * filter[0][2])
      output_pixel = output_pixel + (img[x-1, y] * filter[1][0])
      output_pixel = output_pixel + (img[x, y] * filter[1][1])
      output_pixel = output_pixel + (img[x+1, y] * filter[1][2])
      output_pixel = output_pixel + (img[x-1, y+1] * filter[2][0])
      output_pixel = output_pixel + (img[x, y+1] * filter[2][1])
      output_pixel = output_pixel + (img[x+1, y+1] * filter[2][2])
      output_pixel = output_pixel * weight
      if(output_pixel<0):
        output_pixel=0
      if(output_pixel>255):
        output_pixel=255
      img_transformed[x, y] = output_pixel



plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(img_transformed)
plt.show()