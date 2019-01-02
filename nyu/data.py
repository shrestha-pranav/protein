
# coding: utf-8

# In[2]:


import os
import csv
import tqdm
import numpy as np
import pandas as pd
import pickle as pk


# In[3]:


import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-20s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("data_formatter")


# In[4]:


DATA_DIR = "../data"
DATA_V0  = "../data/v0"


# In[5]:


def readTrainData():
    logger.debug("Reading training data")
    inp_file = pd.read_csv(f"{DATA_DIR}/train_input.csv")
    X = inp_file.iloc[:,0:4].values
    
    with np.load(f"{DATA_DIR}/train_output.npz") as out_file:
        Y = [out_file[key] for key in out_file]
    
    return len(X), X, Y

def readTestData():
    logger.debug("Reading test data")
    inp_file = pd.read_csv(f"{DATA_DIR}/test_input.csv")
    X = inp_file.iloc[:,0:4].values
    return len(X), X

def readTrainDataV0(tag):
    with open(f"{DATA_V0}/train_data_{tag}.pkl", "rb") as file:
        n, X, Y = pk.load(file)
    return n, X, Y

def readTestDataV0():
    with open(f"{DATA_V0}/test_data.pkl", "rb") as file:
        n, X = pk.load(file)
    return n, X


# In[9]:


def pickle_data_close_v0(T = 30):
    """ Reads data from DATA_DIR and writes to DATA_V0 """
    n, X, Y = readTrainData()
    
    # Reduce "ACDEFGHIKLMNPQRSTVWXY" and "BEGHILST"
    D1 = {ch: chr(ord('A')+i) for i, ch in enumerate("ACDEFGHIKLMNPQRSTVWXY")}
    D2 = {ch: chr(ord('A')+i) for i, ch in enumerate("BEGHILST")}

    logger.debug("Formatting training data")
    for k in range(n):
        length = X[k][1]
        str1 = ''
        str2 = ''
        for i in range(length):
            str1 += D1[X[k][2][i]]
            str2 += D2[X[k][3][i]]
        X[k][2], X[k][3] = str1, str2
    
    for k in tqdm.tqdm(range(n)):
        length = X[k][1]
        for i in range(length):
            for j in range(i,length):
                if j - i >= 30 and Y[k][i,j] <= T:
                    Y[k][i,j] = 1.0
                else:
                    Y[k][i,j] = 0.0
    
    n_train = int(n * 0.8)
    n_test  = n - n_train
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    
    logger.debug("Pickling formatted training data")
    path = f"{DATA_V0}"
    if not os.path.exists(path): os.makedirs(path)
    
    with open(f"{DATA_V0}/train_data_close_100.pkl", "wb") as file:
        pk.dump((n,X,Y),file, protocol=0)
    
    with open(f"{DATA_V0}/train_data_close_80.pkl", "wb") as file:
        pk.dump((n_train,X_train,Y_train),file, protocol=0)
    
    with open(f"{DATA_V0}/train_data_close_20.pkl", "wb") as file:
        pk.dump((n_test,X_test,Y_test),file, protocol=0)


# In[10]:


def pickle_data_v0():
    """ Reads data from DATA_DIR and writes to DATA_V0 """
    n, X, Y = readTrainData()
    
    # Reduce "ACDEFGHIKLMNPQRSTVWXY" and "BEGHILST"
    D1 = {ch: chr(ord('A')+i) for i, ch in enumerate("ACDEFGHIKLMNPQRSTVWXY")}
    D2 = {ch: chr(ord('A')+i) for i, ch in enumerate("BEGHILST")}

    logger.debug("Formatting training data")
    for k in range(n):
        length = X[k][1]
        str1 = ''
        str2 = ''
        for i in range(length):
            str1 += D1[X[k][2][i]]
            str2 += D2[X[k][3][i]]
        X[k][2], X[k][3] = str1, str2
    
    n_train = int(n * 0.8)
    n_test  = n - n_train
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    logger.debug("Pickling formatted training data")
    path = f"{DATA_V0}"
    if not os.path.exists(path): os.makedirs(path)
    
    with open(f"{DATA_V0}/train_data_100.pkl", "wb") as file:
        pk.dump((n,X,Y),file, protocol=pk.HIGHEST_PROTOCOL)
    
    with open(f"{DATA_V0}/train_data_80.pkl", "wb") as file:
        pk.dump((n_train,X_train,Y_train),file, protocol=pk.HIGHEST_PROTOCOL)
    
    with open(f"{DATA_V0}/train_data_20.pkl", "wb") as file:
        pk.dump((n_test,X_test,Y_test),file, protocol=pk.HIGHEST_PROTOCOL)
    
    del n, X, Y
    del n_train, X_train, Y_train
    del n_test, X_test, Y_test
    
    n, X = readTestData()
    for k in range(n):
        length = X[k][1]
        str1 = ''
        str2 = ''
        for i in range(length):
            str1 += D1[X[k][2][i]]
            str2 += D2[X[k][3][i]]
        X[k][2], X[k][3] = str1, str2
    
    with open(f"{DATA_V0}/test_data.pkl", "wb") as file:
        pk.dump((n,X),file, protocol=pk.HIGHEST_PROTOCOL)


# In[ ]:


def DataV0_length():
    file = pd.read_csv('Data/data_distance.csv')
    X = file.values
    n = X.shape[0]
    y_avg = np.zeros(691)
    for k in range(n):
        y_avg[k+1] = X[k,3]
    file = open('Data2/V0/y_avg.pickle','wb')
    pk.dump(y_avg,file, protocol=0)
    file.close()

def DataLength():
    n, X, _ = readTrainData()
    MinL = 10000
    MaxL = 0
    for k in range(n):
        MinL = min(MinL, X[k][1])
        MaxL = max(MaxL, X[k][1])
    
    n, X = readTestData()
    for k in range(n):
        MinL = min(MinL, X[k][1])
        MaxL = max(MaxL, X[k][1])
    # MinL = 12, MaxL = 691
    print(MinL,MaxL)

def DataDistance():
    n, _, Y = readTrainDataV0('100')
    D = {}
    for k in range(n):
        print(k,n)
        length = Y[k].shape[0]
        for i in range(length):
            for j in range(i+1,length):
                Dis = j - i
                Val = Y[k][i][j]
                if Dis not in D:
                    D[Dis] = Val, Val, Val, 1
                else:
                    MinV, MaxV, AvgV, Count = D[Dis]
                    MinV = min(MinV,Val)
                    MaxV = max(MaxV,Val)
                    AvgV = AvgV + (Val - AvgV) / (Count + 1)
                    D[Dis] = MinV, MaxV, AvgV, Count + 1
    csvfile = open('./Data/data_distance.csv','w')
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for key in D:
        csvwriter.writerow([key,'{0:.2f}'.format(D[key][0]),            '{0:.2f}'.format(D[key][1]),'{0:.2f}'.format(D[key][2])])
    csvfile.close()


# In[ ]:


def DataV1():
    # read data from Data/V0
    # write data to Data/V1
    path = 'Data/V1'
    if not os.path.exists(path):
        os.makedirs(path)
    for tag in ['100','80','20']:
        n, X, Y = readTrainDataV0(tag)
        csvfile = open('./Data/V1/train_data_'+tag+'.csv','w')
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for k in range(n):
            print(tag,k,n)
            length = X[k][1]
            str1 = X[k][2]
            str2 = X[k][3]
            y = Y[k]
            
            for i in range(length - 1):
                # write string [i:]
                # output string length is at least 2
                _str1 = str1[i:]
                _str2 = str2[i:]
                csvwriter.writerow([length-i, _str1, _str2] + [list(y[i][i:])])
                # reverse string
                csvwriter.writerow([length-i, _str1[::-1], _str2[::-1]] + [list(y[length-1][i:])[::-1]])
        csvfile.close()

def readTrainDataV1(tag):
    filename = 'Data/V0/train_data_'+tag+'.pickle'
    for chunk in pd.read_csv(filename, chunksize=5000, quotechar='|', na_filter = False):
        data = chunk.value
        # do something with data


# In[11]:


#A/21 = "ACDEFGHIKLMNPQRSTVWXY"
#B/8  = "BEGHILST"
if __name__ == '__main__':
    # Input files are train_input.csv, train_output.npz and test_input.csv in DATA_DIR
    pickle_data_v0()
#     pickle_data_close_v0(20)

