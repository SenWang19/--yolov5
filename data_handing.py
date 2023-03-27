"""
处理前：
/test_data  # 全部是测试集的图片  jpeg格式
/train_data # 是由jpeg图片格式和xml格式的label信息 混合放在里面。默认一个jpeg对应一个xml文件

处理后：
/test_data  # 全部是测试集的图片  jpeg格式
/train_data
---Annotations   # 原始训练集给的xml格式的labels
---labels        # 经过处理训练集的txt格式labels
---images        # 用于训练的训练集图片存放
---train.txt, val.txt,test.txt  # 均存放在train_data 目录下
"""
import pandas as pd
import csv
import xml.etree.ElementTree as ET
import os
from os import getcwd
import random
import argparse
import cv2


def data_hand(abs_path):
    # 参数设置

    # abs_path 存储训练集和测试集的根目录地址  如 train_data path: ./hy-tmp/train_data; test_data path: ./hy-tmp/test_data. 则abs_path =./hy-tmp

    classes = ['paper_trash', 'packed_trash', 'snakeskin_bag', 'plastic_trash', 'carton', 'stone_waste', 'sand_waste',
               'foam_trash', 'metal_waste', 'wood_waste']  # 训练的object类别
    ### 开始分割
    # 要对train_data中的文件分开
    file_name_list = os.listdir(abs_path + '/train_data')
    Annotations = '/Annotations'
    Images = '/images'
    sets = ['train', 'val']
    if not os.path.exists(abs_path + '/train_data' + Annotations):
        os.makedirs(abs_path + '/train_data' + Annotations)
    if not os.path.exists(abs_path + '/train_data' + Images):
        os.makedirs(abs_path + '/train_data' + Images)
    for i in file_name_list:
        # print(i[-3:])
        if i[-3:] == 'jpg':
            os.rename(abs_path + '/train_data/' + i, abs_path + '/train_data' + Images + '/' + i)
            # print(1)
        elif i[-3:] == 'xml':
            # print(2)
            os.rename(abs_path + '/train_data/' + i, abs_path + '/train_data' + Annotations + '/' + i)

    def write_csv(file_name, row_image):
        path = str(file_name) + '.csv'
        with open(path, 'a', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(row_image)

    def read_csv(file_name):
        path = str(file_name) + '.csv'
        with open(path, "r") as f:
            csv_read = csv.reader(f)
            for line in csv_read:
                print(line)

    def write_csv_rows(file_name, row_image):
        path = str(file_name) + '.csv'
        with open(path, 'a', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerows(row_image)

    object_image_message = []
    image_message = []
    train_filename_list = []

    # 将获取的图片数据集进行文件名称获取和保存。
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpeg_path', default=abs_path + '/test_data', type=str, help='input test_jpeg label path')
    parser.add_argument('--xml_path', default=abs_path + '/train_data/Annotations', type=str,
                        help='input xml label path')
    parser.add_argument('--txt_path', default=abs_path + '/train_data/main', type=str, help='output txt label path')
    parser.add_argument('--test_path', default=abs_path, type=str, help='output test_jpeg path')
    opt = parser.parse_args(args=[])

    trainval_percent = 1.0
    train_percent = 1.0
    xmlfilepath = opt.xml_path
    txtsavepath = opt.txt_path
    testsavepath = opt.test_path
    total_xml = os.listdir(xmlfilepath)
    jpegfilepath = opt.jpeg_path
    total_jpeg = os.listdir(jpegfilepath)
    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)

    num = len(total_xml)
    list_index = range(num)
    tv = int(num * trainval_percent)
    tr = int(num * train_percent)
    trainval = random.sample(list_index, tv)  # 查一下这个sample
    train = random.sample(trainval, tr)

    file_trainval = open(txtsavepath + '/trainval.txt', 'w')
    file_test = open(abs_path + '/train_data/test.txt', 'w')
    file_train = open(txtsavepath + '/train.txt', 'w')
    file_val = open(txtsavepath + '/val.txt', 'w')
    number = 0
    for i in list_index:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            file_trainval.write(name)
            train_filename_list.append(total_xml[i][:-4] + '.jpg')
            if i in train:
                file_train.write(name)
            else:
                file_val.write(name)
            if number % 3 == 0:
                file_val.write(name)
            number += 1
    for i in range(len(total_jpeg)):
        name = total_jpeg[i] + '\n'
        file_test.write(abs_path + '/test_data/' + name)  # '/hy-tmp/test_data/'+name
    file_trainval.close()
    file_test.close()
    file_train.close()
    file_val.close()

    # 对xml数据进行处理，提取关键的数据。
    def convert(size, box):
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def convert_annotation(image_id):
        in_file = open(abs_path + '/train_data/Annotations/%s.xml' % (image_id),
                       encoding='utf-8')  # ./train_data/Annotations
        out_file = open(abs_path + '/train_data/labels/%s.txt' % (image_id), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        image_message.append([image_id, w, h, w * h])
        for obj in root.iter('object'):
            if obj.find('difficult'):
                difficult = obj.find('difficult').text
            else:
                difficult = 0
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            object_image_message.append([image_id, w, h, w * h, cls_id, b1, b2, b3, b4, (b2 - b1) * (b4 - b3)])
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            if (b2 - b1) * (b4 - b3) > 0:  # 存在object框选范围面积为0的情况
                out_file.write(str(cls_id) + ' ' + ' '.join([str(a) for a in bb]) + '\n')

    wd = getcwd()
    for image_set in sets:
        if not os.path.exists(abs_path + '/train_data/labels/'):
            os.makedirs(abs_path + '/train_data/labels/')
        image_ids = open(abs_path + '/train_data/main/%s.txt' % (image_set)).read().split('\n')[:-1]
        list_file = open(abs_path + '/train_data/%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            list_file.write(abs_path + '/train_data/images/%s.jpg\n' % (image_id))
            # print(image_id)
            convert_annotation(image_id)
        list_file.close()

    # 对上述情况进行处理  修正一些数据。
    new_object_image_message = object_image_message.copy()
    for i in new_object_image_message:
        if i[-1] <= 0:  # object是否有0像素的框图
            # print(i)
            new_object_image_message.pop(new_object_image_message.index(i))
            # print(new_object_image_message.index(i))
            # print('1')
        if i[-2] > i[2]:
            i[-2] = i[2]
            # print('2')
        if i[-4] > i[1]:
            i[-4] = i[1]
            # print('3')
        if i[-3] < 0:
            i[-3] = 0
            # print('4')
        if i[-5] < 0:
            i[-5] = 0
            # print('5')

    # 把想要的信息写入csv文件
    header_object_image_message = ['image_name', 'width', 'height', 'image_area', 'cls_id', 'xmin', 'xmax', 'ymin',
                                   'ymax',
                                   'object_area']
    header_image_message = ['image_name', 'width', 'height', 'image_area']
    # write_csv('image_message',header_image_message)
    # write_csv_rows('image_message',image_message)
    # write_csv('object_image_message',header_object_image_message)
    # write_csv_rows('object_image_message',object_image_message)
    # 对数据进行处理后写入新的文档
    write_csv(abs_path + '/train_data/new_object_image_message', header_object_image_message)
    write_csv_rows(abs_path + '/train_data/new_object_image_message', new_object_image_message)

    df = pd.read_csv(abs_path + '/train_data/new_object_image_message.csv')

    def split_image(src_path, x_start, x_end, y_start, y_end, save_path, cls_id, save_labeledimage_path=None):
        # for file in file_names:
        # src_path 具体图片路径，包含后缀
        img = cv2.imread(src_path)
        size = img.shape[0:2]
        w = size[1]
        h = size[0]
        # print(file, w, h)
        # 每行的高度和每列的宽度
        # 保存切割好的图片的路径，记得要填上后缀，以及名字要处理一下，可以是
        # src_path.split('.')[0] + '_' + str((i+1)*(j+1)) + '.jpg'
        row_start = int((x_start - 0.0) / 1000 * 950)
        row_end = int((w - x_end) / 1000 * 50 + x_end)
        col_start = int((y_start - 0.0) / 1000 * 950)
        col_end = int((h - y_end) / 1000 * 50 + y_end)
        # print(row_start, row_end, col_start, col_end)
        # cv2图片： [高， 宽]
        if ((row_end - row_start) <= 5 * (x_end - x_start)) & ((col_end - col_start) <= 5 * (y_end - y_start)):
            child_img = img[col_start:col_end, row_start:row_end]
            cv2.imwrite(save_path, child_img)
            new_objects = (int((x_start - 0.0) / 1000 * 50), x_end - row_start, int((y_start - 0) / 1000 * 50),
                           y_end - col_start)  # 正常的坐标值，做一下归一化处理
            # print(int((x_start-0.0)/1000*50),x_end-row_start,int((y_start-0)/1000*50),y_end-col_start)
            # show_and_save_image(save_path, int((x_start - 0.0) / 1000 * 50), x_end - row_start,int((y_start - 0) / 1000 * 50), y_end - col_start, save_labeledimage_path)
            new_x, new_y, new_w, new_h = convert((row_end - row_start, col_end - col_start), new_objects)  # 归一化后的值
        else:
            row_start = int(x_start - 2 * (x_end - x_start))
            row_end = int(x_end + 2 * (x_end - x_start))
            col_start = int(y_start - 2 * (y_end - y_start))
            col_end = int(y_end + 2 * (y_end - y_start))
            child_img = img[col_start:col_end, row_start:row_end]
            cv2.imwrite(save_path, child_img)
            new_objects = (int(2 * (x_end - x_start)), int(3 * (x_end - x_start)), int(2 * (y_end - y_start)),
                           int(3 * (y_end - y_start)))  # 正常的坐标值，做一下归一化处理
            # print(int((x_start-0.0)/1000*50),x_end-row_start,int((y_start-0)/1000*50),y_end-col_start)
            # show_and_save_image(save_path,int(2*(x_end-x_start)),int(3*(x_end-x_start)),int(2*(y_end-y_start)),int(3*(y_end-y_start)),save_labeledimage_path)
            # 上面一行是对裁剪的图像画框并另外处理。
            new_x, new_y, new_w, new_h = convert((row_end - row_start, col_end - col_start), new_objects)  # 归一化后的值
        result = [cls_id, new_x, new_y, new_w, new_h]
        return result

    list_file = open(abs_path + '/train_data/train.txt', 'a+')
    for i in range(len(df['image_name'])):
        try:
            if df['object_area'][i] < 6000:
                src_path = abs_path + '/train_data/images/' + df['image_name'][i] + '.jpg'
                save_path = abs_path + '/train_data/images/' + df['image_name'][i] + '_' + str(
                    df['cls_id'][i]) + '_' + str(
                    i) + '.jpg'

                new_objects = split_image(src_path, df['xmin'][i], df['xmax'][i], df['ymin'][i], df['ymax'][i],
                                          save_path,
                                          df['cls_id'][i])  # 归一化后的新的object的值  ,save_labeledimage_path
                list_file.write(abs_path + '/train_data/images/%s.jpg\n' % (
                        df['image_name'][i] + '_' + str(df['cls_id'][i]) + '_' + str(i)))
                f = open(
                    abs_path + '/train_data/labels/' + df['image_name'][i] + '_' + str(df['cls_id'][i]) + '_' + str(
                        i) + '.txt', 'w')
                f.write(' '.join([str(a) for a in new_objects]) + '\n')
                f.close()
        except:
            continue
    list_file.close()
