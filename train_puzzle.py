#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
from dataloader import *
from VGG import *
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet
from keras.layers import Concatenate
from keras.layers.core import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import os
import random
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_PATH = "/mnt/disks/imagenet"
DIM_H = DIM_W = 255
PATCH_SIZE = 75

def load_data(path, val_ratio):
    all_data = os.listdir(path)
    random.shuffle(all_data)
    print "TOTAL DATA: ", len(all_data)
    n_val_files = int(len(all_data)*val_ratio)
    val_files = all_data[:n_val_files]
    train_files = all_data[n_val_files:]
    return train_files[:1000], val_files[:1000]

def PuzzleModel(input_shape=(75,75,27)): #input_shape from data generator
	assert K.image_dim_ordering() == 'tf'
	img_input = Input(shape=input_shape)
    base_model = MobileNet(input_shape=(input_shape[0], input_shape[1], 3), include_top=False)
    print "MOBILENET SUMMARY"
    print base_model.summary()
	patch_outputs = []
	for i in range(9):
		patch_input = img_input[:, :, i*3:(i+1)*3] ##############?
		output = base_model(patch_input)
		print output.get_shape().as_list()
		output_width = output.get_shape().as_list()[-2]
		output = MaxPooling2D((output_width, output_width), name='final_pool')(output)
		output = Flatten(name='flatten')(output)
		patch_outputs.append(output)

	merged_output = Concatenate(patch_outputs, axis=-1)
	final_output = Dense(9, activation='softmax', name='final_fc')(merged_output)
	full_model = Model(inputs=img_input, outputs=final_output)
	return full_model


def train_puzzle():
    train_files, val_files = load_data(path, val_ratio=0.15)
    train_generator = DataLoader(train_files, dim_h=DIM_H, dim_w=DIM_W, batch_size=256, \
                                n_permutations=6, patch_size=PATCH_SIZE).generate()
    val_generator = DataLoader(val_files, dim_h=DIM_H, dim_w=DIM_W, batch_size=256, \
                                n_permutations=1, patch_size=PATCH_SIZE).generate()
    #model = VGG16(input_dim=64, input_depth=27, output_dim=9)
    model = PuzzleModel(input_shape=(PATCH_SIZE, PATCH_SIZE, 3))
    optimizer = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    filepath = "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [es, checkpoint]
    model.fit_generator(generator = train_generator, \
                        steps_per_epoch = train_generator.max_steps, epochs=100, \
                        callbacks=callbacks_list, validation_data = val_generator, \
                        validation_steps = val_generator.max_steps, workers=4)
