import numpy as np

######### section 1 Loading the dataset

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical   


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


######### section 2,  Examining the dataset

print("Shape of training data:")
print(X_train.shape)
print(y_train.shape)
print("Shape of test data:")
print(X_test.shape)
print(y_test.shape)


######### section 3,  matplot of cifar images

#import matplotlib.pyplot as plt
#
#cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#print('Example training images and their labels: ' + str([x[0] for x in y_train[0:5]])) 
#print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))
#
#f, axarr = plt.subplots(1, 5)
#f.set_size_inches(16, 6)
#
#for i in range(5):
#    img = X_train[i]
#    axarr[i].imshow(img)
#plt.show()


######### section 4, preparing the dataset

# Transform label indices to one-hot encoded vectors

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalization of pixel values (to [0-1] range)

X_train /= 255
X_test /= 255


######### section 5, Data Augmentation 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True)   # flip images horizontally

validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train[:40000], y_train[:40000], batch_size=32)
validation_generator = validation_datagen.flow(X_train[40000:], y_train[40000:], batch_size=32)


#train_datagen = ImageDataGenerator(
#        rescale=1./255,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True)
#
#test_datagen = ImageDataGenerator(rescale=1./255)
#
#train_generator = train_datagen.flow_from_directory(
#        'data/train',
#        target_size=(150, 150),
#        batch_size=32,
#        class_mode='binary')
#
#validation_generator = test_datagen.flow_from_directory(
#        'data/validation',
#        target_size=(150, 150),
#        batch_size=32,
#        class_mode='binary')

#model.fit_generator(
#        train_generator,
#        steps_per_epoch=2000,
#        epochs=50,
#        validation_data=validation_generator,
#        validation_steps=800)
#


######### section 5, Adding CNN layers 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3)))
# Batch normalization layer added here
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout layer added here
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256))
# Batch normalization layer added here
model.add(BatchNormalization())
model.add(Activation('relu'))
# Dropout layer added here
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)a
adam = Adam(lr=0.0006, beta_1=0.9, beta_2=0.999, decay=0.0)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

######### section 6, Train the model 

#history = model.fit(X_train, y_train, batch_size=32, epochs=15, verbose=2, validation_split=0.2)

# fits the model on batches with real-time data augmentation:

history = model.fit_generator(train_generator,    
                    validation_data=validation_generator,
                    validation_steps=len(X_train[40000:]) / 32,
                    steps_per_epoch=len(X_train[:40000]) / 32,
                    epochs=15,
                    verbose=1)


######### section 7, Plot training history

#def plotLosses(history):  
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'validation'], loc='upper left')
#    plt.show()
#
#plotLosses(history)


######### section 8, Evaluate the model

score = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
print(model.metrics_names)
print(score)















