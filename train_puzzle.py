#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
from dataloader import *
from VGG import *
import keras

TRAIN_PATH = "/mnt/disks/imagenet/"
VAL_PATH = "/mnt/disks/imagenet/"

def train_puzzle():
    train_generator = DataLoader(TRAIN_PATH, dim_h=225, dim_w=225).generate()
    val_generator = DataLoader(VAL_PATH, dim_h=225, dim_w=225).generate()
    model = VGG16(input_dim=64, input_depth=27, output_dim=9)
    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit_generator(generator = train_generator,
                        steps_per_epoch = train_generator.max_steps, epochs=100,
                        validation_data = val_generator,
                        validation_steps = val_generator.max_steps, workers=4)