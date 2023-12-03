##################################
# Input: V7 exported json file
# Output: dataset folder
#       - folder name
#           - Frames
#           - Labels (single .xml)
#           - Readme.txt
##################################
import os
import argparse
from lxml import etree as ET
import json
import requests

def download_frame(json_name, url, ind, dst, view):
    response = requests.get(url)
    name = '_'.join([json_name[:-5], view, str(ind)]) + '.png'
    path = os.path.join(dst, name)
    with open(path, "wb") as f:
         f.write(response.content)

def to_single_xml(frame_info, frame_folder, label_folder, view):
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = frame_info['folder'][:-5]


    file_name = ET.SubElement(root, "filename")
    file_name.text = '_'.join([frame_info['folder'][:-5], view, frame_info['id']]) + '.png'
    img_path = ET.SubElement(root, "path")
    img_path.text = os.path.join(frame_folder, file_name.text)
    source_ = ET.SubElement(root, "source")
    database_ = ET.SubElement(source_, "database")
    database_.text = "Unknown"

    size_ = ET.SubElement(root, "size")
    width_ = ET.SubElement(size_, "width")
    width_.text = str(frame_info['size'][0])
    height_ = ET.SubElement(size_, "height")
    height_.text = str(frame_info['size'][1])
    depth_ = ET.SubElement(size_, "depth")
    depth_.text = "3"

    segment_ = ET.SubElement(root, "segment")
    segment_.text = "0"

    for loc, name in frame_info['bbox']: # (location, name)
        object_ = ET.SubElement(root, "object")
        name_ = ET.SubElement(object_, "name")
        name_.text = name
        pose_ = ET.SubElement(object_, "pose")
        pose_.text = "Unspecified"
        truncated_ = ET.SubElement(object_, "truncated")
        truncated_.text = "0"
        difficult_ = ET.SubElement(object_, "difficult")
        difficult_.text = "0"

        bndbox_ = ET.SubElement(object_, "bndbox")
        x_min_ = ET.SubElement(bndbox_, "xmin")
        x_min_.text = str(loc['x'])
        y_min_ = ET.SubElement(bndbox_, "ymin")
        y_min_.text = str(loc['y'])
        x_max_ = ET.SubElement(bndbox_, "xmax")
        x_max_.text = str(loc['x'] + loc['w'])
        y_max_ = ET.SubElement(bndbox_, "ymax")
        y_max_.text = str(loc['y'] + loc['h'])

    myfile = '{}/{}.xml'.format(label_folder, file_name.text[:-4])
    et = ET.ElementTree(root)
    et.write(myfile, pretty_print=True)


def v7_json_parser(json_folder, json_name, data_dst, view_name):
    # Input: json file name
    # Output: 
    #   .png in Frames folder
    #   .xml in Labels folder
    #   Readme.txt beside Frames/Labels folder

    with open(os.path.join(json_folder, json_name), 'r') as f:
        data = json.load(f)

    w = data['item']['slots'][0]['width']
    h = data['item']['slots'][0]['height']

    
    frame_count = data['item']['slots'][0]['frame_count']  # not all frames has bbox
    frame_list = data['item']['slots'][0]['frame_urls']

    data_folder = os.path.join(data_dst, json_name[:-5])
    frame_folder = os.path.join(data_folder, 'Frames')
    label_folder = os.path.join(data_folder, 'Labels')

    
    if os.path.exists(data_folder):
        raise Exception('Path {} exists. Delete before regenerate.'.format(data_folder))
    
    os.mkdir(data_folder)
    os.mkdir(frame_folder)
    os.mkdir(label_folder)

    # Labels
    super_dict = {}
    code_stat = {}
    for batch in data['annotations']:
        code = batch['name']
        if code not in code_stat.keys():
            code_stat[code] = 0
        for k, v in batch['frames'].items():
            if k not in super_dict.keys():
                super_dict[k] = {'bbox':[(v['bounding_box'], code)], 'size':(w,h), 'folder':json_name, 'id':k}
            else:
                super_dict[k]['bbox'].append((v['bounding_box'], code))
            code_stat[code] += 1

    for i, k in enumerate(super_dict.keys()):
        print('Generating  V7:{} labels {}/{}...'.format(json_name, i+1, len(frame_list)), end='\r')
        to_single_xml(super_dict[k], frame_folder, label_folder, view_name)
    print('\n')

    # Frames
    count = 1
    for i, furl in enumerate(frame_list):
        if str(i) in super_dict.keys():
            print('Downloading V7:{} frames {}/{}...'.format(json_name, count, len(super_dict)), end='\r')
            download_frame(json_name, furl, ind=i, dst=frame_folder, view=view_name)
            count += 1
    print('\n')

    with open(os.path.join(data_folder, 'Readme.txt'), 'w') as f:
        f.write('Number of Frames in Video: {}\n'.format(frame_count))
        f.write('Number of Frames with BBox: {}\n'.format(len(super_dict)))
        f.write('Code Statistics: {}'.format(code_stat))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True, help='json file from V7')
    parser.add_argument('--view', type=str, required=True, help='view of the video')
    opt = parser.parse_args()

    json_folder = '/home/anonymous/data/ppe_data/'
    json_name = opt.json
    view_name = opt.view
    data_dst = '/home/anonymous/data/ppe_data/'
    v7_json_parser(json_folder, json_name, data_dst, view_name)