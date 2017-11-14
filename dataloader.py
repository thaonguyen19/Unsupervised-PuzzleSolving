import numpy as np 
import os
import random
import multiprocessing as mp
from keras import backend as K
from PIL import Image
import functools

def random_crop(img, dim_h, dim_w):
    h, w = img.shape[0], img.shape[1]
    range_h = (h - self.dim_h) // 2
    range_w = (w - self.dim_w) // 2
    offset_w = 0 if rangew == 0 else np.random.randint(range_w)
    offset_h = 0 if rangeh == 0 else np.random.randint(range_h)
    return img[offset_h:(offset_h+dim_h), offset_w:(offset_w+dim_w), :]

def mean_subtract(img):
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img

class DataGenerator(object):
    def __init__(self, path, dim_h, dim_w, batch_size=256, n_permutations=3, proportion_use=0.5, patch_size=64):
        """
        dim_h, dim_w : dimensions of the 3x3 grid cropped from initial image
        n_permutations: number of different patch permutations of each training image
        proportion_use: proportion of the 1.3M ImageNet data used for training
        patch_size: final size of each patch in the 3x3 grid
        """
    	self.path = path
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.batch_size = batch_size
        self.n_permutations = n_permutations
        self.patch_size = patch_size
        self.max_steps = 0

    def __load_data(self):
        all_classes = os.listdir(self.path)
        for c in all_classes:
            class_path = os.path.join(self.path, c)
            image_files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
            n_selected = round(len(image_files)*proportion_use)
            selected_files = random.sample(image_files, n_selected) 
            all_data += selected_files
        random.shuffle(all_data)
        print "number of training data: " + len(all_data)
        return all_data

    def generate(self):
        all_files = self.__load_data()
        for i in range(self.n_permutations):
            all_files += all_files
        self.max_steps = int(len(all_files)/self.batch_size)
        while True:
            for i in range(self.max_steps):
                selected_files = all_files[i*self.batch_size:(i+1)*self.batch_size]
                X, Y = __generate_data(selected_files)
                yield X, Y

    def __generate_data(self, selected_files): 
        assert K.image_dim_ordering() == 'tf' #ASSUME BACKEND IS TF
        #X = np.empty((self.batch_size, self.dim_h, self.dim_w, 3*9)), Y = np.empty((self.batch_size, 9), dtype = int)
        def load_image(filename):
            try:
                return Image.open(filename).convert('RGB')
            else:
                print "Fail to load %s" % filename

        def generate_patches(img): 
        	'''
        	1 2 3
        	4 5 6
        	7 8 9
        	'''
            patches = []
            h, w = img.shape[0], img.shape[1]
            patch_h, patch_w = h//3, w//3
            assert (patch_h==self.dim_h and patch_w==self.dim_w)
            for h_ind in range(3):
                for w_ind in range(3):
                    patch = img[h_ind*patch_h: (h_ind+1)*patch_h, w_ind*patch_w: (w_ind+1)*patch_w, :]
                    #crop 64x64 patch from 75x75 patch
                    patch_cropped = random_crop(patch, self.patch_size, self.patch_size)
                    patches.append(patch)
            permutation_order = list(range(9))
            random.shuffle(permutation_order)
            shuffled_patches = []
            for i in permutation_order:
                shuffle_patches.append(patches[i])
            return np.dstack(shuffle_patches), permutation_order

        pool = mp.pool(8)
        pil_imgs = pool.map(load_image, selected_files)
        imgs = pool.map(mean_subtract, imgs)
        cropper = functools.partial(random_crop, dim_h=self.dim_h, dim_w=self.dim_w) #225x225 crop
        imgs = pool.map(cropper, pil_imgs) 
        pool.close()
        pool.join()
        patch_list, permute_list = zip(*map(generate_patches, imgs)) 
        #list of dim_h x dim_w x 27 patches from each img file, list of 9-element lists
        Y = np.vstack(permute_list)
        X = np.stack(patch_list)
        return X, Y