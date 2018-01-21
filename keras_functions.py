from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
#from __future__ import print_function
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D,SeparableConv2D
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
#import hbdata_predict_v2 as dataimport
from keras.models import model_from_json
import argparse
#from Main import visualization
# fix random seed for reproducibility
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from keras.layers.normalization import BatchNormalization
from model_functions import Singleton

class cnn_parameters(object):
    __metaclass__ = Singleton
    def __init__(self, weightsPath, model_name, weights, nb_filters, nb_classes, nb_hidden, loss, optimizer, nb_epoch,
                 batch_size, class_fact):

        self.weightsPath = weightsPath
        self.model_name = model_name
        self.weights = weights
        self.nb_filters = nb_filters
        self.nb_classes = nb_classes
        self.nb_hidden = nb_hidden
        self.loss = loss
        self.optimizer = optimizer
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.class_fact = class_fact

class cnn_model(object):
    @staticmethod
    def build (width, height, depth, classes, weightsPath=None,nb_classes=2):
            model = Sequential()
            model.add(Conv2D(filters=80, kernel_size=(10, 10),strides=(2,2), kernel_initializer='orthogonal', padding='valid',
                         kernel_regularizer=l2(0.0001), kernel_constraint=maxnorm(2.),
                         input_shape=(3, 50, 50), data_format='channels_first'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            # dropout to reduce overfitting:
            model.add(Dropout(0.25))
            model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2),data_format='channels_first'))
            model.add(Flatten())
            model.add(Activation('relu'))
            model.add(Dropout(0.25))
            model.add(Dense(nb_classes))
            model.add(Activation('softmax'))
            # if a weights path is supplied (inicating that the model was
            # pre-trained), then load the weights
            if weightsPath is not None:
                model.load_weights(weightsPath)
                print('weight loaded')
            model.summary()

            # return the constructed network architecture
            return model

def proccess_visuallization(history ):
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(history.history['loss'], color='b', label="Training loss")
        ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
        legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
        ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
        legend = ax[1].legend(loc='best', shadow=True)
        plt.show()

        return True


def test_model(X_test,Y_test,
    model_name="model.json",
    weights = 'hb_weights.hdf5',  loss = 'categorical_crossentropy',
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=1e-6)):
    # load json and create model
    json_file = open(file=model_name,mode= 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    prediction = loaded_model.predict(X_test)




    return Y_test,prediction



def train_model(X_train,
                X_val,
                Y_train,
                Y_val,
                load_weights=False,
                model_name="model.json",
                weights = 'hb_weights.hdf5',
                nb_filters= 5,
                nb_classes=2,
                nb_hidden=16,
                loss = 'categorical_crossentropy',
                optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=1e-6),
                nb_epoch=50,
                batch_size=256,
                class_fact=4.):

    #TODO : USE cnn_model class instead singlur values
    model=Sequential()
    model.add(Conv2D(filters=nb_filters, kernel_size=(10, 10),strides=(2,2), kernel_initializer='orthogonal', padding='valid',
                     kernel_regularizer=l2(0.0001), kernel_constraint=maxnorm(2.),
                     input_shape=(3, 50, 50), data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # dropout to reduce overfitting:
    model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2),data_format='channels_first'))

    model.add(Conv2D(filters=nb_filters*2, kernel_size=(3, 3),strides=(2,2), kernel_initializer='orthogonal', padding='valid',
                     kernel_regularizer=l2(0.0001), kernel_constraint=maxnorm(2.),
                     data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # dropout to reduce overfitting:
    model.add(Dropout(0.25))
    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),data_format='channels_first'))

    model.add(Flatten())
   # model.add(Dense(nb_hidden))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
   # model.add(Dense(nb_hidden))
   # model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    if (load_weights):
        model.load_weights(weights)
    model.summary()

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    model_json = model.to_json()
    with open(model_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights)
    print("Saved model to disk")
    checkpointer = ModelCheckpoint(filepath=weights, verbose=1, save_best_only=True,save_weights_only=True)
    class_weight = {0: class_fact,
                    1:1}
    history= model.fit(X_train, Y_train, batch_size=batch_size,class_weight=class_weight,
              nb_epoch=nb_epoch,
              shuffle=True,
              callbacks=[checkpointer],
              verbose=2,
              validation_data=(X_val, Y_val))
    print('Fitting finished, Plotting results...')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()
    prediction = model.predict(X_val)
    #visualization(prediction,Y_val)
    #visualization(1-prediction, 1-Y_val)
    return prediction,Y_val



def  get_feturemaps(model,layer_idx,X_batch):
    get_activation= K.function([model.layers[0].input,K.learning_phase()],[model.layers[layer_idx].output,])
    activation=get_activation([X_batch,0])
    return activation


def buildmodel():
    numpy.random.seed(7)
    # load pima indians dataset
    #dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    #X = dataset[:, 0:8]
    #Y = dataset[:, 8]
    # create model
    normal_folder=''
    abnormal_folder=''
    max_features=5000
    maxlen=100
    batch_size=32
    embedding_dims=100
    nb_filter=30
    hidden_dims=256
    nb_epoch=500
    nb_classes=2
    optimizer='sgd'
    loss='categorical_crossentropy'
    test_split=0.2
    seed=1955
    #model_json='testmodel.json'
    weights='testweights.hdf5'
    load_weights=False
    normal_path=''
    abnormal_path=''

    model=Sequential()
    # We start off with using Convolution2D for a frame
    # The filter is 3x57
    model.add(Conv2D(filters=nb_filter,kernel_size=(3,3),kernel_initializer='orthogonal',padding='valid',
                     kernel_regularizer=l2(0.0001),kernel_constraint=maxnorm(2.),
                     input_shape=(3,50,50),data_format='channels_first' ))
    '''
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
            # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        
        nb_filter=nb_filter,
        nb_row=3,
        nb_col=57,
        init='orthogonal',
        border_mode='valid',
        W_regularizer=l2(0.0001),
        W_constraint=maxnorm(2),
        input_shape=(3, 129,129)))'''

    model.add(Activation('relu'))

        # dropout to reduce overfitting:
    model.add(Dropout(0.25))

        # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling2D(pool_size=(3, 4), strides=(1, 3)))

    # the second convolution layer is 1x3
    '''       #model.add(Conv2D())
    model.add(Conv2D(filters=nb_filter,kernel_size=(1,3),kernel_initializer='orthogonal',padding='valid',kernel_regularizer=l2(0.0001),kernel_constraint=maxnorm(2.)))

    #nb_filter,nb_row=1, b_col=3, init='orthogonal',  W_regularizer=l2(0.0001),W_constraint=maxnorm(2)
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # we use max pooling again:
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    '''
    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())

    # we add two hidden layers:
    # increasing number of hidden layers may increase the accuracy, current number is designed for the competition
    model.add(Dense(hidden_dims,
                    kernel_initializer='Orthogonal',
                    kernel_regularizer=l2(0.0001),
                    kernel_constraint=maxnorm(2)))

    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(hidden_dims,
                    kernel_initializer='Orthogonal',
                    kernel_regularizer=l2(0.0001),
                    kernel_constraint=maxnorm(2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # We project onto a binary output layer to determine the category (Currently: normal/abnormal, but you can try train on the exact abnormality also)
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    #.fit(X, Y, epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    #scores = model.evaluate(X, Y, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    #model.save_weights("model.h5")
    print("Saved model to disk")

    # later...


    return True

