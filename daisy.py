import numpy as np
import cv2
from matplotlib import pyplot as plt

img_path = '/Users/Thao/ILSVRC2012_img_val/n02106382/ILSVRC2012_val_00027219.JPEG'
#img_path = '/Users/Thao/ILSVRC2012_img_val/n03709823/ILSVRC2012_val_00000435.JPEG'
img = cv2.imread(img_path, 0)
#cv2.imshow('image', img)
#cv2.waitKey(0)

orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
print len(kp)
# draw only keypoints location,not size and orientation
img2 = img
img2 = cv2.drawKeypoints(img,kp,img2,color=(0,255,0), flags=0)
#cv2.imshow('image', img2)
#cv2.waitKey(0)

def random_crop(img, dim_h, dim_w):
    h, w = img.shape[0], img.shape[1]
    range_h = (h - dim_h) // 2
    range_w = (w - dim_w) // 2
    offset_w = 0 if range_w == 0 else np.random.randint(range_w)
    offset_h = 0 if range_h == 0 else np.random.randint(range_h)
    return img[offset_h:(offset_h+dim_h), offset_w:(offset_w+dim_w)]

def keypoint_crop(img, dim_h, dim_w):
    h, w = img.shape[0], img.shape[1]
    range_h = (h - dim_h) #// 2
    range_w = (w - dim_w) #// 2
    max_count = float('-inf')
    corner = None
    for offset_w in range(0, range_w):
    	for offset_h in range(0, range_h):
    		end_w = offset_w+dim_w
    		end_h = offset_h+dim_h
    		count = 0
    		for point in kp:
    			h, w = point.pt[0], point.pt[1]
    			if offset_h <= h <= end_h and offset_w <= w <= end_w:
    				count += 1
    		print offset_h, offset_w
    		print count
    		if count >= max_count:
    			max_count = count
    			corner = (offset_h, offset_w)
    ###DEBUG
    print max_count
    offset_h, offset_w = corner
    print corner
    return img[corner[0]:(corner[0]+dim_h), corner[1]:(corner[1]+dim_w)]

img3 = keypoint_crop(img, 255, 255)
cv2.imshow('image', img3)
cv2.waitKey(0)
img4 = random_crop(img, 255, 255)
#cv2.imshow('image', img4)
#cv2.waitKey(0)
