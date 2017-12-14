#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
from dataloader import *
from VGG import *
from keras.optimizers import SGD, Adam
from keras.applications.mobilenet import MobileNet
from keras.layers import Lambda, concatenate, Concatenate
from keras.layers import Input, Reshape, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import os
import random
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from hamming_optimization import *

DATA_PATH = "/lfs/1/ddkang/specializer/imagenet/ILSVRC2012_img_val"
DIM_H = DIM_W = 255
PATCH_SIZE = 76
BATCH_SIZE = 64
loss_hamming = True
RESUME_FILE = None #"weights-15-0.64.hdf5"

def load_data(path, val_ratio):
    all_data = []
    for class_name in os.listdir(path):
        subpath = os.path.join(path, class_name)
        all_data += [os.path.join(subpath, i) for i in os.listdir(subpath)]
    random.shuffle(all_data)
    n_val_files = int(len(all_data)*val_ratio)
    val_files = all_data[:n_val_files]
    train_files = all_data[n_val_files:]
    return train_files, val_files

def PuzzleModel(input_shape=(76,76,27)): #input_shape from data generato
    assert K.image_dim_ordering() == 'tf'
    img_input = Input(shape=input_shape)
    base_model = MobileNet(input_shape=(input_shape[0], input_shape[1], 3), weights=None, include_top=False)
    def slice(img_input, ind):
        return img_input[:, :, :, ind*3:(ind+1)*3]
    patch_outputs = []
    for i in range(9):
        patch_input = Lambda(slice, arguments={'ind':i})(img_input)
        output = base_model(patch_input)
        output_width = output.get_shape().as_list()[-2]
        output = MaxPooling2D((output_width, output_width))(output)
        output_depth = output.get_shape().as_list()[-1]
        output = Reshape((output_depth,))(output)
        patch_outputs.append(output)

    merged_output = Concatenate(axis=-1)(patch_outputs)
    final_output = Dense(64, activation='softmax', name='final_fc')(merged_output)
    full_model = Model(inputs=img_input, outputs=final_output)
    return full_model

def train_puzzle():
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()
    all_permutations = generate_all_permutations()
    if loss_hamming:
        optimal_hamming = find_ave_hamming(all_permutations)
        print optimal_hamming
        all_permutations_arranged = rearrange_cluster(all_permutations)
        if find_ave_hamming(all_permutations_arranged) < optimal_hamming:
            all_permutations = all_permutations_arranged
            optimal_hamming = find_ave_hamming(all_permutations_arranged)
        print optimal_hamming
        all_permutations_arranged = rearrange_greedy(all_permutations)
        if find_ave_hamming(all_permutations_arranged) < optimal_hamming:
            all_permutations = all_permutations_arranged
            optimal_hamming = find_ave_hamming(all_permutations_arranged)
        print optimal_hamming

    train_files, val_files = load_data(DATA_PATH, val_ratio=0.1)
    trainLoader = DataLoader(train_files, all_permutations, dim_h=DIM_H, dim_w=DIM_W, batch_size=BATCH_SIZE, \
                                n_permutations=20, patch_size=PATCH_SIZE)
    train_generator = trainLoader.generate()
    valLoader = DataLoader(val_files, all_permutations, dim_h=DIM_H, dim_w=DIM_W, batch_size=BATCH_SIZE, \
                                n_permutations=2, patch_size=PATCH_SIZE)
    val_generator = valLoader.generate()
    model = PuzzleModel(input_shape=(PATCH_SIZE, PATCH_SIZE, 27))
    filepath = "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    if RESUME_FILE:
        print "RESUME TRAINING ..."
        model.load_weights(RESUME_FILE)
        filepath = "weights-resumed-{epoch:02d}-{val_acc:.2f}.hdf5"
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [es, checkpoint]
    model.fit_generator(generator = train_generator, \
                        steps_per_epoch = trainLoader.max_steps, epochs=100, \
                        callbacks=callbacks_list, validation_data = val_generator, \
                        validation_steps=valLoader.max_steps, max_queue_size=BATCH_SIZE*2, workers=8)#, use_multiprocessing=True)

if __name__ == '__main__':
    train_puzzle()
