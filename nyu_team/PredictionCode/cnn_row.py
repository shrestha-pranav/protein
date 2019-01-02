import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import csv
import tensorflow as tf
from keras import Sequential
#from keras import backend as K
from keras import optimizers
from keras.layers import Convolution2D
from keras.layers import Convolution2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Reshape
#from keras.layers import GlobalMaxPool2D
from keras.models import model_from_json
import pickle as pk
from time import time
import os
import random
import sys
from PIL import Image

DATA_V0 = "../../data/v0"

def CNN_row_1():
    # initialize
    model = Sequential()
    
    # convolution layer
    model.add(Convolution2D(32,kernel_size=(7,8), strides=(2,1), input_shape=(691,8,1),
                                padding='valid',activation='relu'))
    model.add(Convolution2D(64,kernel_size=(7,1), strides=(2,1),
                                padding='valid',activation='relu'))
    
    # max Pooling
    model.add(MaxPooling2D(pool_size=(3,1)))
    
    model.add(Flatten())
    
    model.add(Dense(691,activation='relu'))
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,loss='mse',metrics=['mae'])
    return model

def readTrainDataV0(tag):
    with open(f"{DATA_V0}/train_data_{tag}.pkl", "rb") as file:
        n, X, Y = pk.load(file)
    return n, X, Y

def saveModelTo(model,model_dir, model_name, pool = None, nextID = 0):
    # 1. save model
    model_json = model.to_json()
    with open(model_dir + '/' + model_name + '.json', "w") as json_file:
        json_file.write(model_json)
    # 2. serialize weights to HDF5
    model.save_weights(model_dir + '/' + model_name + '.h5')
    # 3. save pool
    if pool is None:
        pool = np.zeros(3643,dtype=int)
        for i in range(3643):
            pool[i] = i
    file = open(model_dir + '/' + model_name + '.pickle','wb')
    pk.dump((pool, nextID),file, protocol=pk.HIGHEST_PROTOCOL)
    file.close()
    print("Saved model to disk")

def loadModelFrom(model_dir, model_name, learning_rate = 0.1):
    json_file = open(model_dir + '/' + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_dir + '/' + model_name + '.h5')
    print("Loaded model from disk")
    sgd = optimizers.SGD(lr=learning_rate, decay=0, momentum=0.9, nesterov=True)
    loaded_model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
    # load pool
    file = open(model_dir + '/' + model_name + '.pickle','rb')
    pool, nextID = pk.load(file)
    file.close()
    return loaded_model, pool, nextID

def WriteLog(log_filename,logs):
    log_file = open(log_filename,'a')
    log_file.write('\n'.join(logs) + '\n')
    log_file.close()


def FromMatrixToBatchData(x,y):
    '''
    split the matrix into rows of data
    for input length n with n*n matrix
    generate 2n row of data
    '''
    n = x[1] # input length
    
    # Step One: encode origin X
    X_org = np.zeros((n,8,1))
    for k in range(n):
        # for 8-bit encoding
        X_org[k][ord(x[3][k])-ord('A')][0] = 1.0
    
    # Step Two: Fill out X and Y
    X = np.zeros((2*n,691,8,1))
    if y is not None:
        Y = np.zeros((2*n,691))
        #for k in range(2*n):
        #    Y[k] = y_avg
    
    
    for k in range(n):
        # Add seq[k:] staring from k
        start = k
        length = n - start
        X[k][:length] = X_org[start:]
        if y is not None:
            Y[k][:length] = y[start][start:]
        
        # Add seq[:k+1] ends at k
        end = k + 1 # seq[:end]
        length = k + 1
        X[n+k][:length] = X_org[:end][::-1]
        if y is not None:
            Y[n+k][:length] = y[k][:end][::-1]
    if y is not None:
        return 2*n, X, Y/25.0
    else:
        return 2*n, X

def FromPredToMatrix(Y_pred):
    '''
    reconstruct n*n matrix from 2*n row of predictions
    '''
    n = Y_pred.shape[0] // 2
    y = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            y[i][j] = y[j][i] = (Y_pred[i][j-i]+Y_pred[n+j][j-i]) / 2
    return y * 25.0

def Train(model_dir, K, E, learning_rate, testing = True):
    log_filename =  model_dir+'/logs'
    separation_line = '+-------------------------------------------'
    logs = []
    logs.append(separation_line)
    if learning_rate is None:
        logs.append('|   K = {0}, E = {1}'.format(K,E))
    else:
        logs.append('|   K = {0}, E = {1} learning_rate = {2}'.format(K,E,learning_rate))
    logs.append(separation_line)
    logs.append('|             RMSE               TIME')
    WriteLog(log_filename,logs)
    
    # read data
    n_train, X_train, Y_train = readTrainDataV0('80')
    n_test, X_test, Y_test = readTrainDataV0('20')
    
    # load model
    model, pool, nextID = loadModelFrom(model_dir,'model',learning_rate)
    
    for k in range(K):
        print('k = {0}:'.format(k))
        # training
        start = time()
        # train T matrices each time
        T = int(E * n_train)
        for t in range(T):
            # print message
            print('\r training {0} : {1:4d}/{2:4d}'.format(k,t+1,T),end='',flush=True)
            # use sequence-replacement sampling(SRS)
            randID = random.randrange(n_train)
            index = pool[randID]
            pool[randID] = nextID
            nextID = (nextID + 1) % n_train
            # get batch data
            batch_size, X, Y = FromMatrixToBatchData(X_train[index],Y_train[index])
            # fit model
            model.fit(X,Y,batch_size=batch_size,epochs=1,verbose=0)
            #break
        print('')
        end = time()
        train_time = end - start
        
        # save model
        saveModelTo(model,model_dir,'model', pool, nextID)
        saveModelTo(model,model_dir+'/backups','model_'+str(k), pool, nextID)
        
        # test on 80% train data
        # and 20% train data
        if testing:
            start = time()
            RMSE = np.zeros(2)
            SE = np.zeros(n_train)
            for t in range(n_train):
                print('\r testing {0} : {1:4d}/{2:4d}'.format(k,t+1,n_train),end='',flush=True)
                batch_size, X = FromMatrixToBatchData(X_train[t],None)
                Y_pred = FromPredToMatrix(model.predict(X,batch_size=batch_size))
                SE[t] = np.sum((Y_pred - Y_train[t])**2)
                #break
            print('')
            RMSE[0] = np.sqrt(np.mean(SE))
            
            SE = np.zeros(n_test)
            for t in range(n_test):
                print('\r testing {0} : {1:4d}/{2:4d}'.format(k,t+1,n_test),end='',flush=True)
                batch_size, X = FromMatrixToBatchData(X_test[t],None)
                Y_pred = FromPredToMatrix(model.predict(X,batch_size=batch_size))
                SE[t] = np.sum((Y_pred - Y_test[t])**2)
                #break
            print('')
            RMSE[1] = np.sqrt(np.mean(SE))
            end = time()
            test_time = end - start
            
            logs = []
            logs.append('|   {0} : {1:.2f}, {2:.2f} - {3:3.0f} min, {4:3.0f} min'.format(\
                            k,RMSE[0],RMSE[1],train_time/60,test_time/60))
            WriteLog(log_filename,logs)
        else:
            logs = []
            logs.append('|   {0} : Train {1:3.0f} min'.format(k,train_time/60))
            WriteLog(log_filename,logs)
        print('')
        
    logs = []
    logs.append(separation_line+'\n')
    WriteLog(log_filename,logs)
    

if __name__=='__main__':
    # name of model directory
    model_dir = './Models/Row/8_1'
    
    # create model directory if not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(model_dir+'/backups')
        os.makedirs(model_dir+'/Images')
        file = open(model_dir+'/logs','wb')
        file.close()
        model = CNN_row_1()
        saveModelTo(model,model_dir,'model')
        logs = []
        logs.append('+-------------------------------------------')
        logs.append('|                    CNN                   |')
        logs.append('+-------------------------------------------')
        WriteLog(model_dir+'/logs',logs)
    # train model
    Train(model_dir, K = 30, E = 2, learning_rate = 0.1)
    Train(model_dir, K = 15, E = 2, learning_rate = 0.01)
    
    