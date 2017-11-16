import numpy as np 
import multiprocessing as mp
from keras import backend as K
from PIL import Image
import functools
import random
from keras.preprocessing.image import img_to_array
import threading

def random_crop(img, dim_h, dim_w):
    h, w = img.shape[0], img.shape[1]
    range_h = (h - dim_h) // 2
    range_w = (w - dim_w) // 2
    offset_w = 0 if range_w == 0 else np.random.randint(range_w)
    offset_h = 0 if range_h == 0 else np.random.randint(range_h)
    return img[offset_h:(offset_h+dim_h), offset_w:(offset_w+dim_w), :]

def mean_subtract(img):
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator."""
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self
    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

class DataLoader(object):
    def __init__(self, files, dim_h, dim_w, batch_size=256, n_permutations=6, patch_size=64):
        """
        dim_h, dim_w : dimensions of the 3x3 grid cropped from initial image
        n_permutations: number of different patch permutations of each training image
        proportion_use: proportion of the 1.3M ImageNet data used for training
        patch_size: final size of each patch in the 3x3 grid
        """
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.batch_size = batch_size
        self.n_permutations = n_permutations
        self.patch_size = patch_size
        self.files = list(files)
        for i in range(self.n_permutations-1):
            self.files += files
        random.shuffle(self.files)
        print len(self.files)
        print "done preparing files"
        self.max_steps = int(len(self.files)/self.batch_size)
    
    @threadsafe_generator
    def generate(self):
        while True:
            for i in range(self.max_steps):
                selected_files = self.files[i*self.batch_size:(i+1)*self.batch_size]
                X, Y = self.__generate_data(selected_files)
                yield X, Y

    def __generate_data(self, selected_files): 
        assert K.image_dim_ordering() == 'tf' #ASSUME BACKEND IS TF
        #X = np.empty((self.batch_size, self.dim_h, self.dim_w, 3*9)), Y = np.empty((self.batch_size, 9), dtype = int)
        def load_image(filename):
            img = Image.open(filename).convert('RGB')
            resized = False
            if img.size[0] < self.dim_w:
                resized = True
                img = img.resize((self.dim_w,img.size[1]), Image.ANTIALIAS)    
            if img.size[1] < self.dim_h:
                if resized:
                    img = img.resize((self.dim_w, self.dim_h), Image.ANTIALIAS)
                else:
                    img = img.resize((img.size[0], self.dim_h), Image.ANTIALIAS)
            return img

        def generate_patches(img): 
            '''
            1 2 3
            4 5 6
            7 8 9
            '''
            patches = []
            h, w = img.shape[0], img.shape[1]
            patch_h, patch_w = h//3, w//3
            for h_ind in range(3):
                for w_ind in range(3):
                    patch = img[h_ind*patch_h: (h_ind+1)*patch_h, w_ind*patch_w: (w_ind+1)*patch_w, :]
                    #crop 64x64 patch from 75x75 patch
                    patch_cropped = random_crop(patch, self.patch_size, self.patch_size)
                    patches.append(patch_cropped)
            permutation_order = list(range(9))
            random.shuffle(permutation_order)
            shuffled_patches = []
            for i in permutation_order:
                shuffled_patches.append(patches[i])
            return np.dstack(shuffled_patches), permutation_order

        #pool = mp.pool(8)
        pil_imgs = map(load_image, selected_files)
        imgs = map(img_to_array, pil_imgs)
        cropper = functools.partial(random_crop, dim_h=self.dim_h, dim_w=self.dim_w) 
        imgs = map(cropper, imgs)
        imgs = map(mean_subtract, imgs) 
        #pool.close()
        #pool.join()
        patch_list, permute_list = zip(*map(generate_patches, imgs)) 
        #list of dim_h x dim_w x 27 patches from each img file, list of 9-element lists
        Y = np.vstack(permute_list)
        X = np.stack(patch_list)
        return X, Y
