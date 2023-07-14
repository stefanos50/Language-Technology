from sklearn.model_selection import KFold
import DataPreprocessing
import HelperMethods
import Run
from torch.nn import ReLU,Dropout,ELU,LeakyReLU,ReLU6
from torch.nn import LogSoftmax,Sigmoid,Softmax,Tanh


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
K=0
mode = "train"
param = 'Adamax'
model_type = 'FN'

if model_type == 'FN':
    network_architecture = { 'Linear_1': {'out_features' : 250},
                             'BatchNorm1d_1': {},
                             'activation_1': {'function': LeakyReLU()},
                             'Dropout_1': {'function':Dropout(p=0.2)},
                             'Linear_2': {'out_features': 500},
                             'BatchNorm1d_2': {},
                             'activation_2': {'function': LeakyReLU()},
                             'Dropout_2': {'function': Dropout(p=0.2)},
                             'Linear_5': {},
                             'activation_5': {'function': LogSoftmax(dim=1)},
                             }
else:
    network_architecture = { 'Linear_1': {'out_features' : 900},
                             'activation_1': {'function': ELU()},
                             #'Linear_3': {'out_features': 200},
                             #'activation_3': {'function': ELU()},
                            'Linear_7': {'out_features': 500},
                             'activation_2': {'function': ELU()},
                             'Decoder': {},
                             'Linear_33': {'out_features': 900},
                             'activation_33': {'function': ELU()},
                            'Linear_8': {},
                            'activation_8': {'function': ELU()},
                            }

if mode=='train':
    if K>0:
        X, y, classes = DataPreprocessing.get_digits_dataset(0.0, False,model_type=model_type)
        kf = KFold(n_splits=K)
        historyls = []
        timels = []
        test_accuracyls = []
        train_accuracyls = []
        for train_index, test_index in kf.split(X):
            train_data_x, test_data_x = X[train_index], X[test_index]
            train_data_y, test_data_y = y[train_index], y[test_index]

            history, time, train_accuracy, test_accuracy = Run.run(train_data_x, train_data_y, test_data_x, test_data_y,num_classes=classes,arch=network_architecture,model_type=model_type)
            historyls.append(history)
            timels.append(time)
            test_accuracyls.append(test_accuracy)
            train_accuracyls.append(train_accuracy)

        hist_avg = HelperMethods.history_average(historyls)
        HelperMethods.plot_results(hist_avg, sum(train_accuracyls)/len(train_accuracyls), sum(test_accuracyls)/len(test_accuracyls), sum(timels)/len(timels), str(K)+"-Cross-Validation Results")
        print("Average epoch time: " + str(sum(hist_avg['epoch_time']) / len(hist_avg['epoch_time'])))
        HelperMethods.plot_result_single(hist_avg['epoch_time'], "Epoch Time Plot", "Epochs", "Time Elapsed")

        hist_avg['train_accuracy'] = sum(train_accuracyls)/len(train_accuracyls)
        hist_avg['test_accuracy'] =  sum(test_accuracyls)/len(test_accuracyls)
        hist_avg['time'] = sum(timels)/len(timels)
        hist_avg['architecture'] = network_architecture
        hist_avg['param'] = param
        hist_avg['model_type'] = model_type
        HelperMethods.save_history(hist_avg)
    else:
        train_data_x, train_data_y,test_data_x,test_data_y,validation_data_x,validation_data_y, classes = DataPreprocessing.get_digits_dataset(0.1,True,model_type=model_type)
        history, time, train_accuracy, test_accuracy = Run.run(train_data_x, train_data_y, test_data_x, test_data_y,validation_data_x,validation_data_y ,num_classes=classes,arch=network_architecture,model_type=model_type)
        HelperMethods.plot_results(history,train_accuracy,test_accuracy,time,"No Cross-Validation Results")
        print("Average epoch time: "+str(sum(history['epoch_time'])/len(history['epoch_time'])))
        HelperMethods.plot_result_single(history['epoch_time'], "Epoch Time Plot", "Epochs", "Time Elapsed")

        history['train_accuracy'] = train_accuracy
        history['test_accuracy'] =  test_accuracy
        history['time'] = time
        history['architecture'] = network_architecture
        history['param'] = param
        history['model_type'] = model_type
        HelperMethods.save_history(history)

else:
    train_data_x, train_data_y, test_data_x, test_data_y, validation_data_x, validation_data_y, classes = DataPreprocessing.get_digits_dataset(0.1, True, model_type=model_type)
    Run.run(train_data_x, train_data_y, test_data_x, test_data_y,validation_data_x,validation_data_y,num_classes=classes, arch=network_architecture,mode=mode,model_type=model_type)