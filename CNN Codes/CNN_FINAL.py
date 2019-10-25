# Convolutional Neural Network

''' 1. Build the CNN '''

# Import Libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from skimage import transform

# Initialize the CNN
classifier = Sequential()

# Adding Convolutional Layer
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation =  'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding another hidden layer
classifier.add(Convolution2D(32, 3, 3, activation =  'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# If needed, add another hidden layer (Depending on the accuracy)
'''
classifier.add(Convolution2D(64, 3, 3, activation =  'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
'''

# Flattening
classifier.add(Flatten())

# Full Connection
# Hidden Layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))

# Output Layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compile CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


''' 2. Fit the CNN to the images '''

# Image Augmentation - enriching the datasets to avoid overfitting
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_Set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_Set,
                         steps_per_epoch=8000, # Number of images in the training set
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000 # Number of images in test set
                         )


# Predicting a Single Image
'''
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (64, 64, 3))
    np_image = np.expand_dims(np_image, axis = 0)
    return np_image

image = load('cat_or_dog_1.jpg')
classifier.predict(image)
'''