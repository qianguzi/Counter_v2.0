import os, cv2, json, codecs
from random import shuffle

def build_namefile(file_dir):
  f=open(file_dir+'name.txt','w+')
  names = os.listdir(file_dir+'/img')
  shuffle(names)
  for name in names:
    f.writelines(name[:-4]+'\n')
  f.close()


if __name__ == '__main__':
  imread_dir = '/media/jun/data/capdataset/detection/test/'
  boxes_flie = codecs.open(
        imread_dir+'result_test.json', 'r', encoding='utf-8').read()
  img_boxes = json.loads(boxes_flie)
  for img_name in img_boxes:
    boxes = img_boxes[img_name]
    json.dump(boxes, 
            codecs.open(imread_dir+'result/'+img_name+'.json', 'w', encoding='utf-8'),
            separators=(',', ':'), indent=4)
  