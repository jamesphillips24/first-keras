import tensorflow as tf
import os
import keras

training_input_dir = 'archive/dataset/training_set/'
test_input_dir = 'archive/dataset/test_set/'

def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (64, 64))
    image = image/255.0

    return image, label

def get_images_paths_and_labels(input_dir):
    image_paths = []
    labels = []

    class_names = ['cats', 'dogs']
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(input_dir, class_name)
        for filename in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, filename))
            labels.append(label)
    
    return image_paths, labels

training_image_paths, training_labels = get_images_paths_and_labels(training_input_dir)
test_image_paths, test_labels = get_images_paths_and_labels(test_input_dir)

training_dataset = tf.data.Dataset.from_tensor_slices((training_image_paths, training_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))

training_dataset = training_dataset.map(load_and_preprocess_image)
test_dataset = test_dataset.map(load_and_preprocess_image)

batch_size = 64
training_dataset = training_dataset.shuffle(buffer_size=len(training_image_paths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

model = keras.Sequential()  
model.add(keras.Input(shape=(64, 64, 3)))

model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(training_dataset, epochs=10, batch_size=64, validation_data=(test_dataset))

test_loss, test_acc = model.evaluate(training_dataset)
print(test_acc)