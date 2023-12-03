import matplotlib.pyplot as plt
import numpy as np
import json
from collections import Counter
import random
import seaborn as sns

if __name__ == '__main__':

    train_json = '/home/anonymous/data/ppe_data/train.json'  # trppe
    with open(train_json, 'r') as file:
        data = json.load(file)


    # Extract the keys in the data
    keys = list(data.keys())
    ################################################################
    # Number of items in each section
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])
    num_categories = len(data['categories'])
    
    print('#Img:\t{}'.format(num_images))
    print('#Ann:\t{}'.format(num_annotations))
    print('#Cate:\t{}'.format(num_categories))
    print('*'*50)

    # Examples from each section
    example_image = data['images'][random.randint(0, num_images-1)] if num_images > 0 else None
    example_annotation = data['annotations'][random.randint(0, num_annotations-1)] if num_annotations > 0 else None
    example_category = data['categories'][random.randint(0, num_categories-1)] if num_categories > 0 else None

    print('* Example Image:')
    for k, v in example_image.items():
        print('\t{}: {}'.format(k, v))

    print('* Example Anno:')
    for k, v in example_annotation.items():
        print('\t{}: {}'.format(k, v))

    print('* Example Category:')
    for k, v in example_category.items():
        print('\t{}: {}'.format(k, v))
    print('*'*50)

    ################################################################
    # Count the number of annotations for each image
    annotation_counts = Counter(annotation['image_id'] for annotation in data['annotations'])
    # Calculate the average number of annotations per image
    avg_annotations_per_image = sum(annotation_counts.values()) / num_images

    print('Ave Annos per Image: ', avg_annotations_per_image)
    ################################################################
    # Count the number of annotations for each category
    category_counts = Counter(annotation['category_id'] for annotation in data['annotations'])

    # Get the category names
    category_names = {category['id']: category['name'] for category in data['categories']}

    # Map category counts to category names
    category_counts_named = {category_names[category_id]: count for category_id, count in category_counts.items()}

    # Get image sizes
    image_sizes = [(image['width'], image['height']) for image in data['images']] # all (1920, 1080)
    annotation_area = [anno['area']/(1920*1080) for anno in data['annotations']]

    # Count the number of annotations for each image
    annotation_counts_values = list(annotation_counts.values())
    print('Counts in each category')
    for k, v in category_counts_named.items():
        print('\t{}: {}'.format(k, v))
    print('*'*50)

    ################################################################
    # colors
    rgb_black = [i/255 for i in [0, 18, 25]]
    rgb_green = [i/255 for i in [9, 147, 150]]
    rgb_orange = [i/255 for i in [238, 155, 0]]
    rgb_red = [i/255 for i in [174, 32, 18]]
    rgb_light_green = [i/255 for i in [145, 211, 192]]
    rgb_dark_green = [i/255 for i in [0, 96, 115]]

    sns.set_style("whitegrid")
    ############################## 1 ##################################
    # Plotting distribution of annotations across categories
    fig_title = 'anno_distribution'
    name_nickname = ['MA', 'MI', 'RC', 'NC', 'S', 'GA', 'GI', 'EA', 'FC', 'FI', 'SG', 'PG', 'GG', 'PR', 'HA', 'HC', 'Head'] 
    bar_x = {}
    for name in list(category_counts_named.keys()):
        if name == 'Head':
            bar_x['Head'] = name
        else:
            sub_name = name.split('_')[-1].upper()
            bar_x[sub_name] = name

    print(bar_x)
    bar_y = [category_counts_named[bar_x[x]] for x in name_nickname]

    # VERSON 1 - Vertical
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=name_nickname, y=bar_y, color=rgb_light_green, edgecolor=rgb_black)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2., p.get_height(), '{0:0.0f}'.format(p.get_height()), 
            fontsize=16, color='black', ha='center', va='bottom')
    plt.xlabel('Category', fontsize=22)
    plt.ylabel('Number of annotations', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig('{}_vertical.png'.format(fig_title))

    # VERSON 2 - Horizontal
    plt.figure(figsize=(12, 15))
    ax = sns.barplot(y=name_nickname, x=bar_y, color=rgb_light_green, edgecolor=rgb_black)
    for i, v in enumerate(bar_y):
        ax.text(v, i, " "+'{:,}'.format(v), va='center', fontsize=26, color='black', ha='left')
    plt.ylabel('Category', fontsize=28)
    plt.xlabel('Number of annotations', fontsize=28)
    plt.yticks(fontsize=26)
    plt.xticks(fontsize=26)
    plt.tight_layout()
    plt.savefig('{}_horizontal.png'.format(fig_title))
    ############################# 2 ###################################
    bbox_aspect_ratios = [bbox[2] / bbox[3] for bbox in (annotation['bbox'] for annotation in data['annotations'])]
    bbox_aspect_ratios_min = np.min(bbox_aspect_ratios)
    bbox_aspect_ratios_max = np.max(bbox_aspect_ratios)
    bbox_aspect_ratios_mean = np.mean(bbox_aspect_ratios)
    bbox_aspect_ratios_median = np.median(bbox_aspect_ratios)
    
    print('=> Bbox w/h ratios MIN, MAX, MEAN, MEDIAN')
    print(bbox_aspect_ratios_min, bbox_aspect_ratios_max, bbox_aspect_ratios_mean, bbox_aspect_ratios_median)
    
    # Plotting distribution of bounding box aspect ratios
    fig_title = 'bbox_size_ratio'
    plt.figure(figsize=(12, 7))
    sns.histplot(bbox_aspect_ratios, bins=50, color=rgb_light_green, edgecolor=rgb_black)
    plt.xlabel('Bounding box aspect ratio (width / height)', fontsize=28)
    plt.ylabel('Number of bounding boxes', fontsize=28)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig('{}.png'.format(fig_title))

    ############################# 3 ###################################
    # Plot distribution of annotation area
    fig_title = 'area_distribution'
    plt.figure(figsize=(12, 7))
    num_bin = 20
    ax = sns.histplot(annotation_area, bins=num_bin, kde=False, color=rgb_light_green, edgecolor=rgb_black)
    _, bin_edges = np.histogram(annotation_area, bins=num_bin)
    ax.set_xticks(bin_edges)
    plt.xlabel('Area Ratio', fontsize=28)
    plt.ylabel('Number of Bounding Box', fontsize=28)
    plt.xticks(fontsize=22, rotation=45)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig('{}.png'.format(fig_title))