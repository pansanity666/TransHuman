import cv2
import numpy as np
import glob
import os
import sys

path = sys.argv[1] 

files = os.listdir(path)
files.sort(key=lambda x: int(x.split('.')[0]))
print(files)

speed = 30
img_array = []
# for idx in range(len(files)):
for f in files:
    print(path + "/" + f)
    img = cv2.imread(path + "/" + f)

    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('/'.join(path.split('/')) + '_demo.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), speed, size)

for i in range(len(img_array)):
    print(img_array[i].shape)
    out.write(img_array[i])
out.release()
