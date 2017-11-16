#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
from dataloader import *
from VGG import *
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet
from keras.layers import Lambda, concatenate, Concatenate
from keras.layers import Input, Reshape, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import os
import random
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_PATH = "/home/thachthao2605/imagenet/ILSVRC2012_img_val"
DIM_H = DIM_W = 255
PATCH_SIZE = 76

def load_data(path, val_ratio):
    all_data = [os.path.join(path, i) for i in os.listdir(path)]
    random.shuffle(all_data)
    n_val_files = int(len(all_data)*val_ratio)
    val_files = all_data[:n_val_files]
    train_files = all_data[n_val_files:]
    return train_files, val_files

def PuzzleModel(input_shape=(75,75,27)): #input_shape from data generato
    assert K.image_dim_ordering() == 'tf'
    img_input = Input(shape=input_shape)
    base_model = MobileNet(input_shape=(input_shape[0], input_shape[1], 3), weights=None, include_top=False)
    def slice(img_input, ind):
        return img_input[:, :, :, ind*3:(ind+1)*3]
    patch_outputs = []
    for i in range(9):
        #print "IMAGE INPUT:", img_input.get_shape().as_list()
        patch_input = Lambda(slice, arguments={'ind':i})(img_input)
        #patch_input = img_input[:, :, :, i*3:(i+1)*3] ##############?
        #print "PATCH SHAPE: ", patch_input.get_shape().as_list()
        output = base_model(patch_input)
        #print "OUTPUT FROM MOBILE NET: ", output.get_shape().as_list()
        output_width = output.get_shape().as_list()[-2]
        output = MaxPooling2D((output_width, output_width))(output)
        #print "AFTER MAXPOOLING: ", output.get_shape().as_list()
        #output = Flatten(name='flatten')(output)
        output_depth = output.get_shape().as_list()[-1]
        output = Reshape((output_depth,))(output)
        patch_outputs.append(output)

    #merged_output = concatenate(patch_outputs, axis=-1)
    merged_output = Concatenate(axis=-1)(patch_outputs)
    #print "MERGED OUTPUT SHAPE: ", merged_output.get_shape().as_list()
    final_output = Dense(9, activation=None, name='final_fc')(merged_output)
    full_model = Model(inputs=img_input, outputs=final_output)
    return full_model

def train_puzzle():
    train_files, val_files = load_data(DATA_PATH, val_ratio=0.1)
    trainLoader = DataLoader(train_files, dim_h=DIM_H, dim_w=DIM_W, batch_size=64, \
                                n_permutations=18, patch_size=PATCH_SIZE)
    train_generator = trainLoader.generate()
    valLoader = DataLoader(val_files, dim_h=DIM_H, dim_w=DIM_W, batch_size=64, \
                                n_permutations=2, patch_size=PATCH_SIZE)
    val_generator = valLoader.generate()
    model = PuzzleModel(input_shape=(PATCH_SIZE, PATCH_SIZE, 27))
    optimizer = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    filepath = "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [es, checkpoint]
    model.fit_generator(generator = train_generator, \
                        steps_per_epoch = trainLoader.max_steps, epochs=100, \
                        callbacks=callbacks_list, validation_data = val_generator, \
                        validation_steps=valLoader.max_steps, max_queue_size=64, workers=8)#, use_multiprocessing=True)

if __name__ == '__main__':
    train_puzzle()
