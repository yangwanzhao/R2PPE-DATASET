######################################################
# This is to calculate density that to evaluate how 
# crowded an image is with the bounding boxes.
# Input: coco format annotation json file
# Output: density 
######################################################
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

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

def density_1(anno_by_img):
    density_1_per_bbox = {}
    nums = 1
    for image_id, annotations in anno_by_img.items():
        print('* Density#1 {}/{}'.format(nums, len(anno_by_img)), end='\r')
        bboxes = [annotation['bbox'] for annotation in annotations]
        density_1_per_bbox[image_id] = []
        for i in range(len(bboxes)):
            ious = [compute_iou(bboxes[i], bboxes[j]) for j in range(len(bboxes)) if j != i]
            density_1 = np.mean(ious) if ious else 0
            density_1_per_bbox[image_id].append(density_1)
        nums += 1
    print('')

    # Compute the average density#1 per image
    avg_density_1_per_image = {image_id: np.mean(densities) for image_id, densities in density_1_per_bbox.items()}

    # Show the average density#1 per image
    avg_density_1_per_image_values = list(avg_density_1_per_image.values())

    return avg_density_1_per_image_values

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

    return avg_density_2_per_image_values

def density_3(anno_by_img):
    density_3_per_image = {}
    nums = 1
    for image_id, annotations in anno_by_img.items():
        print('* Density#3 {}/{}'.format(nums, len(anno_by_img)), end='\r')
        bboxes = [annotation['bbox'] for annotation in annotations]
        total_overlap_area = 0
        total_bbox_area = 0
        for i in range(len(bboxes)):
            total_bbox_area += bboxes[i][2] * bboxes[i][3]
            # Note: Same area could be counted multiple times
            for j in range(i + 1, len(bboxes)):
                _, overlap_area = compute_iou(bboxes[i], bboxes[j], want_interArea=True)
                total_overlap_area += overlap_area
        density_3 = total_overlap_area / total_bbox_area if total_bbox_area > 0 else 0
        density_3_per_image[image_id] = density_3
        nums += 1
    print('')

    # Show the density#3 per image
    density_3_per_image_values = list(density_3_per_image.values())
    
    return density_3_per_image_values

def density_4(anno_by_img, iou_threshold=0.3):
    density_4_per_image = {}
    nums = 1
    for image_id, annotations in anno_by_img.items():
        print('* Density#4 {}/{}'.format(nums, len(anno_by_img)), end='\r')
        bboxes = [annotation['bbox'] for annotation in annotations]
        num_overlapping_bboxes = 0
        for i in range(len(bboxes)):
            for j in range(len(bboxes)):
                if i != j and compute_iou(bboxes[i], bboxes[j]) >= iou_threshold:
                    num_overlapping_bboxes += 1
                    break
        density_4 = num_overlapping_bboxes / len(bboxes) if len(bboxes) > 0 else 0
        density_4_per_image[image_id] = density_4
        nums += 1
    print('')

    # Show the density#4 per image
    density_4_per_image_values = list(density_4_per_image.values())
    
    return density_4_per_image_values

def global_density(anno):
    annotation_counts = Counter(annotation['image_id'] for annotation in anno['annotations'])
    return list(annotation_counts.values())

if __name__ == '__main__':
    data_json = {'TRPPE': '/home/anonymous/data/ppe_data/train.json',
                 'COCO' : '/home/anonymous/data/COCO2017/instances_train2017.json',
                 'CPPE-5': '/home/anonymous/data/CPPE-5/annotations/train.json',
                 'VOC'  : '/home/anonymous/data/VOC/VOCtrainval_0712-COCO.json'
                }

    locden2compare = []
    glbden2compare = []
    name2compare = []
    for name, path in data_json.items():
        print('Calculation for {}'.format(name))
        with open(path, 'r') as file:
            data = json.load(file)

        annotations_by_image = defaultdict(list)
        for annotation in data['annotations']:
            annotations_by_image[annotation['image_id']].append(annotation)

        print('* Total Number of Images: {}'.format(len(annotations_by_image)))

        # loc_den = density_1(annotations_by_image)
        loc_den = density_2(annotations_by_image)
        # loc_den = density_3(annotations_by_image)
        # loc_den = density_4(annotations_by_image)

        glb_den = global_density(data)

        locden2compare.append(np.array(loc_den))
        glbden2compare.append(np.array(glb_den))
        if name != 'TRPPE':
            name2compare.append(name)
        else:
            name2compare.append('R2PPE')
    
        print('{} Global Density: {}'.format(name, sum(glb_den)/len(glb_den)))

    sns.set_style("whitegrid")

    # colors
    rgb_black = [i/255 for i in [0, 18, 25]]
    rgb_green = [i/255 for i in [9, 147, 150]]
    rgb_orange = [i/255 for i in [238, 155, 0]]
    rgb_red = [i/255 for i in [174, 32, 18]]
    color2compare = {'R2PPE': rgb_red, 'COCO': rgb_orange, 'CPPE-5': rgb_green, 'VOC': rgb_black}

    ############################## 1 ##################################
    # local density
    data_df = pd.DataFrame(locden2compare).T
    data_df.columns = name2compare
    plt.figure(figsize=(10, 6))
    for col in data_df.columns:
        sns.kdeplot(data_df[col], fill=True, color=color2compare[col])

    plt.xlim(-0.1, 0.6)
    plt.xlabel("Local Density", fontsize=26)
    plt.ylabel("Kernel Density Estimate", fontsize=26)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(title='Datasets', loc='upper right', labels=data_df.columns, fontsize=22, title_fontsize=24)
    plt.tight_layout()
    plt.savefig('local_density.png')

    ############################## 2 ##################################
    # global density
    data_df = pd.DataFrame(glbden2compare).T
    data_df.columns = name2compare
    plt.figure(figsize=(10, 6))
    for col in data_df.columns:
        sns.kdeplot(data_df[col], fill=True, color=color2compare[col])

    plt.xlim(-5, 30)
    plt.xlabel("Global Density", fontsize=26)
    plt.ylabel("Kernel Density Estimate", fontsize=26)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(title='Datasets', loc='upper right', labels=data_df.columns, fontsize=22, title_fontsize=24)
    plt.tight_layout()
    plt.savefig('global_density.png')