##############################################################
# This file convert coco format to yolo format 
# output is named 'yolo_labels' saved at the current folder
# note that converting via ultralytics cannot specify the output name but only 'yolo_labels'
# please rename it after 
##############################################################
from ultralytics.yolo.data.converter import convert_coco
import json
import os
import shutil

children_path = '/home/anonymous/data/ppe_data/'

print('=> Start to conver labels')
convert_coco(labels_dir=children_path, cls91to80=False)

# copy images
print('=> Start to copy images')
train_json = os.path.join(children_path, 'train.json')
test_json = os.path.join(children_path, 'test.json')
train_list = []
test_list = []
with open(train_json) as f:
    data = json.load(f)
for img in data['images']:
    train_list.append(img['file_name'])

with open(test_json) as f:
    data = json.load(f)
for img in data['images']:
    test_list.append(img['file_name'])

os.mkdir('train')
os.mkdir('test')

src_data_list = list(os.listdir('/home/anonymous/data/ppe_data/images'))
num_img_removed = 0
for ind, img in enumerate(src_data_list):
    print('Copying Image {}/{}'.format(ind, len(src_data_list)), end='\r')
    src = '/home/anonymous/data/ppe_data/images/' + img
    if img in train_list:
        dst = 'images/train/' + img
        shutil.copyfile(src, dst)
    elif img in test_list:
        dst = 'images/test/' + img
        shutil.copyfile(src, dst)
    else:
        num_img_removed += 1
        
print('Number of Imgs not included: ', num_img_removed)


