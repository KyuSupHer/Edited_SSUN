# -*- coding: utf-8 -*-
"""
@author: Yonghao.Xu
"""


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from HyperFunctions import*

from keras.datasets import mnist
from keras.models import Model,Sequential,save_model,load_model
from keras.layers import Input, Dense, Activation,LSTM,merge,Convolution2D, MaxPooling2D,AveragePooling2D, Flatten,Dropout,add, concatenate
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import np_utils

#MCNN
def MCNN_RS(num_PC, img_rows, img_cols):
    CNNInput = Input(shape=[num_PC, img_rows, img_cols], name='CNNInput')

    CONV1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='CONV1')(CNNInput)
    POOL1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='POOL1')(CONV1)
    CONV2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='CONV2')(POOL1)
    POOL2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='POOL2')(CONV2)
    CONV3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='CONV3')(POOL2)
    POOL3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th", name='POOL3')(CONV3)

    FLATTEN0 = Flatten(name='FLATTEN1')(POOL3)
    # FLATTEN1 = Flatten(name='FLATTEN1')(POOL1)
    # FLATTEN2 = Flatten(name='FLATTEN2')(POOL2)
    # FLATTEN3 = Flatten(name='FLATTEN3')(POOL3)

    DENSE0 = Dense(128, activation='relu', name='DENSE1')(FLATTEN0)
    # DENSE1 = Dense(128,activation='relu', name='DENSE1')(FLATTEN1)
    # DENSE2 = Dense(128,activation='relu', name='DENSE2')(FLATTEN2)
    # DENSE3 = Dense(128,activation='relu', name='DENSE3')(FLATTEN3)

    "CNNDense = merge([DENSE1, DENSE2, DENSE3], mode='sum', name='CNNDense')"
    # CNNDense = add([DENSE1, DENSE2, DENSE3])
    # CNNDense = add([DENSE0])

    CNNSOFTMAX = Dense(nb_classes, activation='softmax', name='CNNSOFTMAX')(DENSE0)

    model = Model(input=[CNNInput], output=[CNNSOFTMAX])
    rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=1e-05)

    model.compile(optimizer=rmsp, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Spatial
w = 30
num_PC = 4
israndom = True
randtime = 1
OASpatial_Pavia = np.zeros((9 + 2, randtime))
s1s2 = 1
time_step = 1

for r in range(0, randtime):
    #################Pavia#################
    dataID = 1
    data = HyperspectralSamples(dataID=dataID, timestep=time_step, w=w, num_PC=num_PC, israndom=israndom, s1s2=s1s2)
    X = data[0]
    X_train = data[1]
    X_test = data[2]
    XP = data[3]
    XP_train = data[4]
    XP_test = data[5]
    Y = data[6] - 1
    Y_train = data[7] - 1
    Y_test = data[8] - 1

    batch_size = 128

    nb_classes = Y_train.max() + 1
    nb_epoch = 500
    nb_features = X.shape[-1]

    img_rows, img_cols = XP.shape[-1], XP.shape[-1]
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(Y_train, nb_classes)
    y_test = np_utils.to_categorical(Y_test, nb_classes)

    model = MCNN_RS(num_PC, img_rows, img_cols)

    histloss = model.fit([XP_train], [y_train], nb_epoch=nb_epoch, batch_size=batch_size, verbose=0, shuffle=True)
    losses = histloss.history

    PredictLabel = model.predict([XP_test], verbose=0).argmax(axis=-1)

    OA, Kappa, ProducerA = CalAccuracy(PredictLabel, Y_test[:, 0])
    OASpatial_Pavia[0:9, r] = ProducerA
    OASpatial_Pavia[-2, r] = OA
    OASpatial_Pavia[-1, r] = Kappa
    print('rand', r + 1, 'MCNN Pavia test accuracy:', OA * 100)

    Map = model.predict([XP], verbose=0)

    Spatial = Map.argmax(axis=-1)

    X_result = DrawResult(Spatial, 1)

    plt.imsave('CNN_only_w30' + repr(r + 1) + 'OA_' + repr(int(OA * 10000)) + '.png', X_result)