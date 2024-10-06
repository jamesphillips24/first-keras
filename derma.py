import tensorflow as tf
from sklearn.utils import class_weight
import os
import keras
import pandas as pd
import numpy as np

data = pd.read_csv("train_labels.csv")
test_data = pd.read_csv("test_labels.csv")
image_files = os.listdir("train")
test_image_files = os.listdir("test")
key = ['benign', 'malignant']

images = []
labels = []
test_images = []
test_labels = []
for f in image_files:
  exists = data[data['isic_id'] == f[:-4]]['benign_malignant'].values
  if exists.size > 0 and pd.notna(exists[0]):
    images.append(os.path.join("train", f))
    labels.append(key.index(exists[0]))
  
for f in test_image_files:
  exists = test_data[test_data['isic_id'] == f[:-4]]['benign_malignant'].values
  if exists.size > 0 and pd.notna(exists[0]):
    test_images.append(os.path.join("test", f))
    test_labels.append(key.index(exists[0]))

def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image/255.0
    #image = tf.expand_dims(image, axis=0)
    return image, label

training_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
training_dataset = training_dataset.map(load_and_preprocess_image)
training_dataset = training_dataset.shuffle(buffer_size=len(images)).batch(64).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.map(load_and_preprocess_image)
test_dataset = test_dataset.shuffle(buffer_size=len(test_images)).batch(64).prefetch(tf.data.AUTOTUNE)

#prediction, l = load_and_preprocess_image('S_0418_Benign_mole_M2200096.width-320.jpg',1)
#model = keras.models.load_model('my_model.keras')
#test_loss, test_acc = model.evaluate(test_dataset)
#print(test_acc)
#result = model.predict(prediction)
#print(result)


class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(labels),
                                                  y=labels)
class_weights_dict = dict(enumerate(class_weights))

base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)
base_model.trainable = False
model = keras.Sequential()
model.add(base_model)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(2, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(training_dataset, epochs=20, batch_size=64, class_weight=class_weights_dict, validation_data=(test_dataset))

test_loss, test_acc = model.evaluate(test_dataset)
print(test_acc)

base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(training_dataset, epochs=5, batch_size=64, validation_data=(test_dataset))

test_loss, test_acc = model.evaluate(test_dataset)
print(test_acc)

model.save("trained_model2.h5")"""


"""model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))"""
