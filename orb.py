import numpy as np
import cv2
from PIL import Image

def random_crop(img, dim_h, dim_w):
    h, w = img.shape[0], img.shape[1]
    range_h = (h - dim_h) // 2
    range_w = (w - dim_w) // 2
    offset_w = 0 if range_w == 0 else np.random.randint(range_w)
    offset_h = 0 if range_h == 0 else np.random.randint(range_h)
    return img[offset_h:(offset_h+dim_h), offset_w:(offset_w+dim_w)]

def keypoint_crop(img, dim_h, dim_w, kp):
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
    			pt_h, pt_w = point.pt[0], point.pt[1]
    			if offset_h <= pt_h <= end_h and offset_w <= pt_w <= end_w:
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
    #for point in kp:
    #	h, w = point.pt[0], point.pt[1]
    #	if offset_h <= h <= (offset_h+dim_h) and offset_w <= w <= (offset_w+dim_w):
    #		#print "%d <= %d <= %d" % (offset_h, h, offset_h+dim_h)
    # 		#print "%d <= %d <= %d" % (offset_w, w, offset_w+dim_w)
    return corner

def keypoint_central_crop(img, dim_h, dim_w, kp):
    h, w = img.shape[0], img.shape[1]
    pt_x = sum([p.pt[0] for p in kp])
    pt_x = float(pt_x)/len(kp)
    pt_y = sum([p.pt[1] for p in kp])
    pt_y = float(pt_y)/len(kp)
    #print "MEAN X, Y: ", pt_x, pt_y
    if pt_x <= w/2.0:
        left_hor_margin = pt_x
        left_hor_margin = min(dim_w/2.0, left_hor_margin)
    else:
        right_hor_margin = w - pt_x
        right_hor_margin = min(dim_w/2.0, right_hor_margin)
        left_hor_margin = dim_w - right_hor_margin
    offset_w = pt_x - left_hor_margin

    if pt_y <= h/2.0:
        upper_ver_margin = pt_y
        upper_ver_margin = min(dim_h/2.0, upper_ver_margin)
    else:
        lower_ver_margin = h - pt_y
        lower_ver_margin = min(dim_h/2.0, lower_ver_margin)
        upper_ver_margin = dim_h - lower_ver_margin
    offset_h = pt_y - upper_ver_margin

    corner = (int(round(offset_h)), int(round(offset_w)))
    #print corner
    return corner


def orb_crop(detector, img_array, dim_h, dim_w):
    img_pil = Image.fromarray(img_array)
    img_grayscale = img_pil.convert(mode="L")
    kp = detector.detect(img,None)
    corner = keypoint_central_crop(img, dim_h, dim_w, kp)
    return img_array[corner[0]:(corner[0]+dim_h), corner[1]:(corner[1]+dim_w), :]

if __name__ == '__main__':
    img_path = '/Users/Thao/ILSVRC2012_img_val/n02106382/ILSVRC2012_val_00027219.JPEG'
    #img_path = '/Users/Thao/ILSVRC2012_img_val/n03709823/ILSVRC2012_val_00000435.JPEG'
    img = cv2.imread(img_path, 0)
    print type(img)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    print len(kp)
    # draw only keypoints location,not size and orientation
    img2 = img
    img2 = cv2.drawKeypoints(img,kp,img2,color=(0,255,0), flags=0)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)

    corner = keypoint_central_crop(img, 255, 255, kp)
    img3 = img[corner[0]:(corner[0]+255), corner[1]:(corner[1]+255)] 
    cv2.imshow('img3', img3)
    cv2.waitKey(0)

    img_array = Image.open(img_path).convert('RGB')
    img_array = np.asarray(img_array)
    img4 = orb_crop(orb, img_array, 255, 255)
    #img4 = Image.fromarray(img4)
    #img4.save('test3.png')

    img5 = random_crop(img, 255, 255)
    cv2.imshow('img5', img5)
    cv2.waitKey(0)
