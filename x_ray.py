import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

BATCH_SIZE = 20

image_train_generator = ImageDataGenerator(
  rescale = 1.0/255,
  height_shift_range = 0.05,
  width_shift_range = 0.05
)
image_validation_generator = ImageDataGenerator(
  rescale = 1.0/255
)
training_iterator = image_train_generator.flow_from_directory(
  'augmented-data/train',
  class_mode = 'categorical',
  color_mode = 'grayscale',
  target_size = (1000, 1000),
  batch_size = BATCH_SIZE
)
validation_iterator = image_train_generator.flow_from_directory(
  'augmented-data/train',
  class_mode = 'sparse',
  color_mode = 'grayscale',
  target_size = (1000, 1000),
  batch_size = BATCH_SIZE
)

model = Sequential()
model.add(keras.Input(shape=(1000, 1000, 1)))
model.add(tf.keras.layers.Conv2D(5, 5, strides=3, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(5, 5), strides=(5,5)))
model.add(tf.keras.layers.Conv2D(6, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(3, activation = 'softmax'))

model.summary()

model.compile(
    optimizer= keras.optimizers.Adam(learning_rate=0.1),
    loss= keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.AUC()]
)
model.fit(
         training_iterator,
         steps_per_epoch=training_iterator.samples/BATCH_SIZE,
         epochs=10,
         validation_data=validation_iterator,
         validation_steps=validation_iterator.samples/BATCH_SIZE)
# Do Matplotlib extension below

# use this savefig call at the end of your graph instead of using plt.show()
# plt.savefig('static/images/my_plots.png')
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

# used to keep plots from overlapping
fig.tight_layout()

fig.savefig('static/images/my_plots.png')

