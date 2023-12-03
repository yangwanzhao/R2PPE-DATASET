#######################################################
# This is to generate trppe dataset in coco format
# path: /home/anonymous/data/ppe_data/PPE_TR-PPE/label_coco
#######################################################
import os
import json
import random
import numpy as np
from copy import deepcopy
from collections import defaultdict

PATH_ROOT = '/home/anonymous/data/ppe_data/PPE_TR-PPE'
PATH_JSON = os.path.join(PATH_ROOT, 'label_v7')
PATH_DATA = os.path.join(PATH_ROOT, 'images')
PATH_COCO = os.path.join(PATH_ROOT, 'label_coco')

def compute_iou(bbox1, bbox2, want_interArea=False):
        """Computes the Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            bbox1_area = w1 * h1
            bbox2_area = w2 * h2
            iou = inter_area / (bbox1_area + bbox2_area - inter_area)
        else:
            inter_area = 0.0
            iou = 0.0

        if want_interArea:
            return iou, inter_area
        else:
            return iou

def density_2(anno_by_img):
    density_2_per_bbox = {}
    nums = 1
    num_neighbors = 1  # number of nearest neighbors to consider
    for image_id, annotations in anno_by_img.items():
        print('* Density#2 {}/{}'.format(nums, len(anno_by_img)), end='\r')
        bboxes = [annotation['bbox'] for annotation in annotations]
        density_2_per_bbox[image_id] = []
        for i in range(len(bboxes)):
            ious = sorted([(j, compute_iou(bboxes[i], bboxes[j])) for j in range(len(bboxes)) if j != i],
                        key=lambda x: -x[1])
            top_ious = ious[:num_neighbors]
            density_2 = np.mean([iou for _, iou in top_ious]) if top_ious else 0
            density_2_per_bbox[image_id].append(density_2)
        nums += 1
    print('')

    # Compute the average density#2 per image
    avg_density_2_per_image = {image_id: np.mean(densities) for image_id, densities in density_2_per_bbox.items()}

    # Show the average density#2 per image
    avg_density_2_per_image_values = list(avg_density_2_per_image.values())

    return avg_density_2_per_image_values, avg_density_2_per_image

if __name__ == '__main__':
    os.makedirs(PATH_COCO)

    # obtain list of images globally
    split_ratio = 0.8 # 0.8
    all_data = list(os.listdir(PATH_DATA))  # images local id in their names starts with 0
    total_imgs = len(all_data)
    total_train = int(total_imgs * split_ratio)
    total_test = total_imgs - total_train
    print('Total Number of Images: {}'.format(total_imgs))
    print('Total Number of Train : {}'.format(total_train))
    print('Total Number of Test  : {}'.format(total_test))

    random.shuffle(all_data) # TODO: set the random seed
    train_data = deepcopy(all_data[:total_train]) 
    test_data = deepcopy(all_data[total_train:]) 
    train_data.sort(key=lambda x:(''.join(x.split('_')[:-1]), int(x.split('_')[-1][:-4])))
    test_data.sort(key=lambda x:(''.join(x.split('_')[:-1]), int(x.split('_')[-1][:-4])))
    data_tuple = [train_data, test_data]

    data_repo = sorted(list(os.listdir(PATH_JSON)))

    train_json = {'images':[], 'annotations': [], 'categories': []}
    test_json = {'images':[], 'annotations': [], 'categories': []}
    label_tuple = [train_json, test_json]

    # 17 classes; Order is important!
    category_all = ['Gown_Absent_ga',
                'Gown_Incomplete_gi',
                'Glove_Absent_ha',
                'Glove_Complete_hc',
                'Regular_Mask_Complete_rc',
                'Mask_Absent_ma',
                'Mask_Incomplete_mi',
                'N95_complete_nc',
                'N95_Strap_s',
                'Face_Shield_Complete_fc',
                'Face_Shield_Incomplete_fi', 
                'Eyewear_Absent_ea',
                'Safety_Glasses_sg',
                'Prescription_Glasses_pg',
                'Goggles_gg',
                'PAPR_pr',
                'Head']

    cate_exclude = []

    category = [i for i in category_all if i not in cate_exclude]
    #################################################################################
    cate_dic_train = dict([(i, 0) for i in category])
    cate_dic_test = dict([(i, 0) for i in category])
    img_exist = [set(), set()] 

    for ind, ca in enumerate(category):
        train_json['categories'].append({'supercategory': None, 'id': ind+1, 'name': ca})
        test_json['categories'].append({'supercategory': None, 'id': ind+1, 'name': ca})

    for dir in data_repo:
        print('Processing JSON {}'.format(os.path.join(PATH_JSON, dir)))
        
        with open(os.path.join(PATH_JSON, dir), 'r') as f:
            ppe_data = json.load(f)

        for batch in ppe_data['annotations']:
            code = batch['name']
            code = code.replace('(', '')
            code = code.replace(')', '')
            code = code.replace(' ', '_')
            assert code in category_all, 'Wrong code: {}'.format(code)
            
            if code not in category:
                continue
            else:
                cate_id = category.index(code) + 1 

            for k, v in batch['frames'].items():
                img_name = '_'.join([dir[:-5]+'_above', k])+'.png'
                if img_name in train_data: 
                    label_index = 0 # train img
                elif img_name in test_data: 
                    label_index = 1 # test img
                else: 
                    raise Exception('Wrong Data List')
                imgID = data_tuple[label_index].index(img_name) + 1 # all ids starts with 1
                img_exist[label_index].add(imgID)
 
                h = max(int(v['bounding_box']['h']), 0)
                w = max(int(v['bounding_box']['w']), 0)
                x = max(int(v['bounding_box']['x']), 0)
                y = max(int(v['bounding_box']['y']), 0)
                if w == 0 or h == 0:
                    continue
                label_dict = {'area': w*h, 
                              'bbox': [x, y, w, h],
                              'category_id': cate_id,
                              'iscrowd': 0,  
                              'ignore': 0,
                              'segmentation': [],
                              'image_id': imgID,
                              'id': len(label_tuple[label_index]['annotations'])+1}
                label_tuple[label_index]['annotations'].append(label_dict)
                if label_index == 0:
                    cate_dic_train[code] += 1
                elif label_index == 1:
                    cate_dic_test[code] += 1
                else:
                    raise Exception('Wrong Label Index!')
                    
        # all images has the same h and w
        width = ppe_data['item']['slots'][0]['width']
        height = ppe_data['item']['slots'][0]['height']

    # TRAIN IMG
    # fill out train img
    for iin, img in enumerate(train_data):
        if iin+1 in img_exist[0]:
            img_dict = {'file_name': img,
                        'height': height,
                        'width': width,
                        'isdense': 0,
                        'id': iin+1}
            train_json['images'].append(img_dict)

    # calculate isdense for train img
    annotations_by_image = defaultdict(list)
    for annotation in train_json['annotations']:
        annotations_by_image[annotation['image_id']].append(annotation)
    print('* Total Number of Images in Train set: {}'.format(len(annotations_by_image)))
    _, img_den = density_2(annotations_by_image)
    cc = 0
    for im in train_json['images']:
        if img_den[im['id']] > 0.29:
            im['isdense'] = 1
            cc += 1
    print(cc)

    # TEST IMG
    # fill out test img
    for iin, img in enumerate(test_data):
        if iin+1 in img_exist[1]:
            img_dict = {'file_name': img,
                        'height': height,
                        'width': width,
                        'isdense': 0,
                        'id': iin+1}
            test_json['images'].append(img_dict)

    # calculate isdense for test img
    annotations_by_image = defaultdict(list)
    for annotation in test_json['annotations']:
        annotations_by_image[annotation['image_id']].append(annotation)
    print('* Total Number of Images in Test set: {}'.format(len(annotations_by_image)))
    _, img_den = density_2(annotations_by_image)
    cc = 0
    for im in test_json['images']:
        if img_den[im['id']] > 0.29:
            im['isdense'] = 1
            cc += 1
    print(cc)

    path = os.path.join(PATH_COCO, 'train.json')
    with open(path, "w") as json_file:
        json.dump(train_json, json_file)
    print("Train JSON file saved successfully under {}.".format(path))
    path = os.path.join(PATH_COCO, 'test.json')
    with open(path, "w") as json_file:
        json.dump(test_json, json_file)
    print("Test JSON file saved successfully under {}.".format(path))

    print('Annos distribution:')
    print('Train:\n', cate_dic_train)
    print('Test:\n', cate_dic_test)
    print('Number of Imgs after Processing:')
    print('Train:\n', len(train_json['images']))
    print('Test:\n', len(test_json['images']))