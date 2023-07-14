import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.utils import shuffle


def convert_to_one_hot_encode(labels):
    return LabelBinarizer().fit_transform(labels)

def plot_class_balance(y):
    from yellowbrick.target import ClassBalance
    # Instantiate the visualizer
    visualizer = ClassBalance(labels=['0','1','2','3','4','5','6','7','8','9'])
    visualizer.fit(y)
    visualizer.show()


def get_digits_dataset(test_size,split=True,model_type='FN'):
    #X = np.load('data.npy')
    #y = np.load('labels.npy')
    X_train = np.load('data.npy')
    y_train = np.load('labels.npy')
    X_test = np.load('data_test.npy')
    y_test = np.load('labels_test.npy')
    X_val = np.load('data_val.npy')
    y_val = np.load('labels_val.npy')
    num_dif_classes = 10
    if model_type == 'FN' or model_type=='RBM':
        y_train = convert_to_one_hot_encode(y_train)
        y_test = convert_to_one_hot_encode(y_test)
        y_val = convert_to_one_hot_encode(y_val)
        #y = convert_to_one_hot_encode(y)
    #X = np.expand_dims(X, axis=1)
    #X = np.reshape(X, (X.shape[0],X.shape[1]*X.shape[2]))
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1] * X_val.shape[2]))

    if model_type == "AE":
        num_dif_classes = X_train.shape[1]

    if split:
        #X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.13, random_state=1)
        return X_train,y_train,X_test,y_test,X_val,y_val,num_dif_classes
    else:
        X = np.concatenate((X_train,X_test,X_val),axis=0)
        y = np.concatenate((y_train,y_test,y_val),axis=0)
        return X,y,num_dif_classes

def get_digits_dataset_single(test_size,split=True,model_type='FN'):
    X = np.load('data_single.npy')
    y = np.load('labels_single.npy')
    plot_class_balance(y)

    num_dif_classes = 10
    if not model_type == 'RBM':
        y = convert_to_one_hot_encode(y)
    #X = np.expand_dims(X, axis=1)
    X = np.reshape(X, (X.shape[0],X.shape[1]*X.shape[2]))

    if model_type == "AE":
        num_dif_classes = X.shape[1]

    if split:
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.13, random_state=1)
        return X_train,y_train,X_test,y_test,None,None,num_dif_classes
    else:
        return X,y,num_dif_classes
