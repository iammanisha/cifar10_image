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

#my code starts
X_train_0 = np.empty((10000,32,32,3),dtype='uint8')
X_train_1 = np.empty((10000,32,32,3),dtype='uint8')
X_train_2 = np.empty((10000,32,32,3),dtype='uint8')
X_train_3 = np.empty((10000,32,32,3),dtype='uint8')
y_train_0 = np.empty((10000,1),dtype='uint8')
y_train_1 = np.empty((10000,1),dtype='uint8')
y_train_2 = np.empty((10000,1),dtype='uint8')
y_train_3 = np.empty((10000,1),dtype='uint8')

for i in range(10000):
    X_train_0[i,:] = X_train[4*i,:]
    X_train_1[i,:] = X_train[4*i+1,:]
    X_train_2[i,:] = X_train[4*i+2,:]
    X_train_3[i,:] = X_train[4*i+3,:]
    y_train_0[i] = y_train[4*i]
    y_train_1[i] = y_train[4*i+1]
    y_train_2[i] = y_train[4*i+2]
    y_train_3[i] = y_train[4*i+3]

######### section 4, preparing the dataset

# Transform label indices to one-hot encoded vectors

X_train_0 = X_train_0.astype('float32')
X_train_1 = X_train_1.astype('float32')
X_train_2 = X_train_2.astype('float32')
X_train_3 = X_train_3.astype('float32')
X_train_0 /= 255
X_train_1 /= 255
X_train_2 /= 255
X_train_3 /= 255
y_train_0 = to_categorical(y_train_0, num_classes=10)
y_train_1 = to_categorical(y_train_1, num_classes=10)
y_train_2 = to_categorical(y_train_2, num_classes=10)
y_train_3 = to_categorical(y_train_3, num_classes=10)

#My code ends
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

train_generator_0 = train_datagen.flow(X_train_0[:10000], y_train_0[:10000], batch_size=32)
train_generator_1 = train_datagen.flow(X_train_1[:10000], y_train_1[:10000], batch_size=32)
train_generator_2 = train_datagen.flow(X_train_2[:10000], y_train_2[:10000], batch_size=32)
train_generator_3 = train_datagen.flow(X_train_3[:10000], y_train_3[:10000], batch_size=32)
validation_generator_0 = validation_datagen.flow(X_train_0[10000:], y_train_0[10000:], batch_size=32)
validation_generator_1 = validation_datagen.flow(X_train_1[10000:], y_train_1[10000:], batch_size=32)
validation_generator_2 = validation_datagen.flow(X_train_2[10000:], y_train_2[10000:], batch_size=32)
validation_generator_3 = validation_datagen.flow(X_train_3[10000:], y_train_3[10000:], batch_size=32)


######### section 5, Adding CNN layers 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

def model_definition(train_generator,validation_generator):
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
                        validation_steps=len(X_train[10000:]) / 32,
                        steps_per_epoch=len(X_train[:10000]) / 32,
                        epochs=1,
                        verbose=1)

model_definition(train_generator_0,validation_generator_0)
model_definition(train_generator_1,validation_generator_1)
model_definition(train_generator_2,validation_generator_2)
model_definition(train_generator_3,validation_generator_3)

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















