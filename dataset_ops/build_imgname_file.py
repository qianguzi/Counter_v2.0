import os, cv2
from random import shuffle

if __name__ == '__main__':
  file_dir = '/media/jun/data/capdataset/detect/train_256/'
  f=open(file_dir+'name.txt','w+')
  names = os.listdir(file_dir+'/img')
  shuffle(names)
  for name in names:
    f.writelines(name[:-4]+'\n')
  f.close()