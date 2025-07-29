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

new_size_x , new_size_y = (int)(size_x/2),(int)(size_y/2)

img_pooled = np.zeros((new_size_x,new_size_y))

for x in range (0,size_x,2):
    for y in range(0,size_y,2):
        pixels = []
        pixels.append(img_transformed[x,y])
        pixels.append(img_transformed[x+1,y])
        pixels.append(img_transformed[x,y+1])
        pixels.append(img_transformed[x+1,y+1])
        pixels.sort(reverse=True)
        img_pooled[(int)(x/2),(int)(y/2)]=pixels[0]
        

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(img_pooled)
plt.show()