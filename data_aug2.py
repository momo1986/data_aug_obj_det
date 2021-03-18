import os
import time
import json
import cv2
import shutil
import random
import numpy as np
from math import *
import xml.etree.ElementTree as ET


CLASS_MAP = {'pedestrian':0, 'cyclist':1, 'car':2}


class DataAug:

    def __init__(self):
        pass

    def copy_files(self, src, dst):
        for file in os.listdir(src):
            old_file=os.path.join(src,file)
            new_file=os.path.join(dst,file)
            shutil.copy(old_file,new_file)

    
    def aug_ps(self, data_path, aug_path, orientation = 'up'):
        '''
            垂直透视
        '''
        x=0
        for label_name in os.listdir(data_path):
            # if label_name.endswith("json"):
            if label_name.endswith("xml"):
                x+=1
                img_name = label_name.split('.')[0]+'.png'
                img = cv2.imread(os.path.join(data_path, img_name))
                H, W = img.shape[:2]         
                
                # 透视
                pts = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
                m=random.randint(5,15)

                if orientation == 'up':
                    pts_ = np.float32([[m, 0], [W - 1 - m, 0], [W - 1, H - 1], [0, H - 1]])
                else:
                    pts_ = np.float32([[0, 0], [W - 1, 0], [W - 1 - m, H - 1], [m, H - 1]])

                # 透视变换矩阵
                M = cv2.getPerspectiveTransform(pts, pts_)
                img_ = cv2.warpPerspective(img, M, (W,H))
                
                # 修改xml文件
                tree = ET.parse(os.path.join(data_path, label_name))
                root = tree.getroot()
                Objects = root.findall('object')

                for Object in Objects:
                    xyxy = Object.find('bndbox')
                    points=[[int(xyxy.find('xmin').text), int(xyxy.find('ymin').text)], 
                            [int(xyxy.find('xmax').text), int(xyxy.find('ymax').text)]]
                    pt_list = [np.array([pt[0], pt[1], 1]).reshape(3,1) for pt in points]

                    new_points = []
                    for pt in pt_list:
                        res = np.matmul(M, pt)
                        res /= res[2][0]
                        res = res.astype(np.int)
                        new_points.append([int(res[0][0]), int(res[1][0])])

                    xyxy.find('xmin').text = str(new_points[0][0])
                    xyxy.find('ymin').text = str(new_points[0][1])
                    xyxy.find('xmax').text = str(new_points[1][0])
                    xyxy.find('ymax').text = str(new_points[1][1])

                
                tree.write(os.path.join(aug_path, label_name.split('.')[0]+'_ps.xml'), encoding='utf-8')

                cv2.imwrite(os.path.join(aug_path, img_name.split('.')[0]+'_ps.png'), img_, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                if x%100 == 0:
                    lt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(lt+" 已透视 %d/%d 张图片"%(x,len(os.listdir(data_path))/2))
    

    def aug_hf(self, data_path, aug_path):
        '''
            水平翻转
        '''
        x=0
        for label_name in os.listdir(data_path):
            if label_name.endswith("xml"):
                x+=1
                img_name = label_name.split('.')[0]+'.png'
                img = cv2.imread(os.path.join(data_path, img_name))
                H, W = img.shape[:2]         
                
                # 水平翻转
                img_center = np.array(img.shape[:2])[::-1] / 2
                img_center = np.hstack((img_center, img_center))
                img_ = img[:, ::-1, :]

                # 修改xml文件
                tree = ET.parse(os.path.join(data_path, label_name))
                root = tree.getroot()
                Objects = root.findall('object')

                for Object in Objects:
                    xyxy = Object.find('bndbox')
                    xyxy.find('xmin').text = str(W - int(xyxy.find('xmin').text))
                    xyxy.find('xmax').text = str(W - int(xyxy.find('xmax').text))

                tree.write(os.path.join(aug_path, label_name.split('.')[0]+'_hf.xml'), encoding='utf-8')

                cv2.imwrite(os.path.join(aug_path, img_name.split('.')[0]+'_hf.png'), img_, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                if x%100 == 0:
                    lt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(lt+" 已翻转 %d/%d 张图片"%(x,len(os.listdir(data_path))/2))


    def rotate(self, img, degree):
        height, width = img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width//2, height//2), degree, 1)
    
        # 加入平移操作
        matRotation[0,2] += (widthNew - width)//2
        matRotation[1,2] += (heightNew - height)//2
    
        imgRotation = cv2.warpAffine(img, matRotation,(widthNew,heightNew),borderValue=(0,0,0))
    
        return imgRotation, matRotation

    def aug_rt(self, data_path, aug_path):
        '''
            旋转
        '''
        x=0
        for label_name in os.listdir(data_path):
            if label_name.endswith("xml"):
                x+=1
                img_name = label_name.split('.')[0]+'.png'
                img = cv2.imread(os.path.join(data_path, img_name))        
                
                # 随机旋转5°或者357°
                if x%2==0:
                    img_, mat = self.rotate(img, 5)
                else:
                    img_, mat = self.rotate(img, 357)
                
                # 修改xml文件
                tree = ET.parse(os.path.join(data_path, label_name))
                root = tree.getroot()
                Objects = root.findall('object')

                for Object in Objects:
                    xyxy = Object.find('bndbox')
                    points=[[int(xyxy.find('xmin').text), int(xyxy.find('ymin').text)], 
                            [int(xyxy.find('xmax').text), int(xyxy.find('ymax').text)]]

                    new_points = [np.dot(mat, np.array([point[0],point[1],1])).tolist()[:2] for point in points]
                    new_points = [[int(p[0]), int(p[1])] for p in new_points]

                    xyxy.find('xmin').text = str(new_points[0][0])
                    xyxy.find('ymin').text = str(new_points[0][1])
                    xyxy.find('xmax').text = str(new_points[1][0])
                    xyxy.find('ymax').text = str(new_points[1][1])

                
                tree.write(os.path.join(aug_path, label_name.split('.')[0]+'_rt.xml'), encoding='utf-8')

                cv2.imwrite(os.path.join(aug_path, img_name.split('.')[0]+'_rt.png'), img_, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                if x%100 == 0:
                    lt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(lt+" 已旋转 %d/%d 张图片"%(x,len(os.listdir(data_path))/2))


    def anisotropy(self, src_img, k2=100, la=0.25, N=20):
        '''
        各向异性扩散滤波
        '''
        def helper(src_img, k2=100, la=0.25):
            src = src_img.copy().astype(np.float32)
            dst = src_img.copy().astype(np.float32)
            rows,cols=src.shape[:2]

            src_w = src[0:rows-2, 1:cols-1] - src[1:rows-1, 1:cols-1]
            src_e = src[2:rows, 1:cols-1] - src[1:rows-1, 1:cols-1]
            src_n = src[1:rows-1, 0:cols-2] - src[1:rows-1, 1:cols-1]
            src_s = src[1:rows-1, 2:cols] - src[1:rows-1, 1:cols-1]

            src_w*=np.exp(-src_w**2/k2)
            src_e*=np.exp(-src_e**2/k2)
            src_n*=np.exp(-src_n**2/k2)
            src_s*=np.exp(-src_s**2/k2)

            src_wens = (src_w + src_e + src_n + src_s)*la

            src_delta = np.zeros(src.shape)
            src_delta[1:rows-1, 1:cols-1] = src_wens

            dst+=src_delta
            dst = dst.astype(np.uint8)
            return dst
        dst = src_img.copy().astype(np.float32)
        for _ in range(N):
            dst = helper(dst)
        return dst


    def aug_an(self, data_path, aug_path):
        x=0
        for label_name in os.listdir(data_path):
            if label_name.endswith("xml"):
                x+=1
                # 对1/8的数据进行滤波处理
                if x%8 == 0:
                    img_name = label_name.split('.')[0]+'.png'
                    img = cv2.imread(os.path.join(data_path, img_name))
                    img_ = self.anisotropy(img)
                    cv2.imwrite(os.path.join(aug_path, img_name.split('.')[0]+'_an.png'), img_, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    shutil.copyfile(os.path.join(data_path, label_name), os.path.join(aug_path, label_name.split('.')[0]+'_an.xml'))
                if x%100 == 0:
                    lt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(lt+" 已滤波处理 %d/%d 张图片"%(x,len(os.listdir(data_path))/2))


    def aug(self, base_path, base_path_aug):
        if os.path.exists(base_path_aug):
            shutil.rmtree(base_path_aug)
        os.makedirs(base_path_aug)

        aug_path_list = []
        aug_path_ext = ['hf', 'rt', 'ps', 'an']
        for ext in aug_path_ext:
            path = base_path_aug + os.sep + ext
            os.makedirs(path)
            aug_path_list.append(path)

        print("==============================================================\n水平翻转增广中...")
        self.aug_hf(base_path, aug_path_list[0])
        self.copy_files(base_path,aug_path_list[0])

        print("==============================================================\n旋转增广中...")
        self.aug_rt(aug_path_list[0], aug_path_list[1])
        self.copy_files(aug_path_list[0],aug_path_list[1])

        print("==============================================================\n透视增广中...")
        self.aug_ps(aug_path_list[1], aug_path_list[2])
        self.copy_files(aug_path_list[1],aug_path_list[2])

        print("==============================================================\n滤波处理增广中...")
        self.aug_an(aug_path_list[2], aug_path_list[3])
        self.copy_files(aug_path_list[2],aug_path_list[3])

        print("==============================================================\n清理中间文件中...")

        for aug_path in aug_path_list:
            self.copy_files(aug_path, base_path_aug)
            shutil.rmtree(aug_path)
        
        print("==============================================================\n已完成增广")

    
    def genTXT(self, base_path_aug, base_path_txt):
        '''
            生成txt文件
        '''
        if os.path.exists(base_path_txt):
            shutil.rmtree(base_path_txt)
        os.makedirs(base_path_txt)

        x=0
        for label_name in os.listdir(base_path_aug):
            # if label_name.endswith("json"):
            if label_name.endswith("xml"):
                x+=1
                
                tree = ET.parse(os.path.join(base_path_aug, label_name))
                root = tree.getroot()

                # 得到宽高
                HW = root.find('size')
                W = int(HW.find('width').text)
                H = int(HW.find('height').text)

                Objects = root.findall('object')

                for Object in Objects:
                    
                    category = CLASS_MAP[Object.find('name').text]

                    xyxy = Object.find('bndbox')
                    xmin = int(xyxy.find('xmin').text)
                    ymin = int(xyxy.find('ymin').text)
                    xmax = int(xyxy.find('xmax').text)
                    ymax = int(xyxy.find('ymax').text)

                    # class x_center y_center width height format
                    x_c = (xmin+xmax)/2/W
                    y_c = (ymin+ymax)/2/H
                    w = (xmax-xmin)/W
                    h = (ymax-ymin)/H

                    with open(os.path.join(base_path_txt, label_name.split('.')[0]+'.txt'), 'a') as txt:
                        txt.write(str(category) + ' ' + str(x_c) + ' ' + str(y_c) + ' ' + str(w) + ' ' + str(h) + '\n')

                if x%100 == 0:
                    lt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(lt+" 已转换 %d/%d 张图片"%(x,len(os.listdir(base_path_aug))/2))



if __name__ == "__main__":
    dataAug = DataAug()

    # 下面三个path根据实际情况修改
    # base_path 包括原始数据集的所有图片和xml标注
    base_path = './data/images'

    # base_path_aug 自动生成，包括增广后的所有图片和xml标注
    base_path_aug = '/media/qiu/新加卷/yolo_data/base_path_aug'

    # base_path_txt 自动生成，包括所有从xml转成txt的标注
    base_path_txt = '/media/qiu/新加/yolo_data/base_path_txt'
    
    # 增广
    dataAug.aug(base_path, base_path_aug)
    
    # 转换标注
    dataAug.genTXT(base_path_aug, base_path_txt)
    

