from sklearn.model_selection import KFold
import HelperMethods
import Run_RBM
import DataPreprocessing


mode = 'eval'
param = 'adam'
hidden_layers = [800,600,500] #800,600,500 - 900,700,500 , 900,500
K=0
if mode=='train':
    if K>0:
        X, y, classes = DataPreprocessing.get_digits_dataset(0.0, False,autoencoder=False,model_type='RBM')
        kf = KFold(n_splits=K)
        historyls = []
        timels = []
        test_accuracyls = []
        train_accuracyls = []
        test_original_accuracyls = []
        train_original_accuracyls = []
        for train_index, test_index in kf.split(X):
            train_data_x, test_data_x = X[train_index], X[test_index]
            train_data_y, test_data_y = y[train_index], y[test_index]

            history, time, train_accuracy_original,  test_accuracy_original, train_accuracy ,  test_accuracy = Run_RBM.run(train_data_x, train_data_y, test_data_x, test_data_y,num_classes=classes,n_hin=hidden_layers)
            historyls.append(history)
            timels.append(time)
            test_accuracyls.append(test_accuracy)
            train_accuracyls.append(train_accuracy)
            test_original_accuracyls.append(test_accuracy_original)
            train_original_accuracyls.append(train_accuracy_original)

        hist_avg = HelperMethods.history_average(historyls)
        HelperMethods.plot_results(hist_avg, sum(train_accuracyls) / len(train_accuracyls),
                                   sum(test_accuracyls) / len(test_accuracyls), sum(timels) / len(timels),
                                   str(K) + "-Cross-Validation Results")
        print("Average epoch time: " + str(sum(hist_avg['epoch_time']) / len(hist_avg['epoch_time'])))
        HelperMethods.plot_result_single(hist_avg['epoch_time'], "Epoch Time Plot", "Epochs", "Time Elapsed")

        hist_avg['train_accuracy'] = round((sum(train_accuracyls) / len(train_accuracyls))*100,2)
        hist_avg['test_accuracy'] = round((sum(test_accuracyls) / len(test_accuracyls))*100,2)
        hist_avg['train_accuracy_original'] = round((sum(train_original_accuracyls) / len(train_original_accuracyls))*100,2)
        hist_avg['test_accuracy_original'] =  round((sum(test_original_accuracyls) / len(test_original_accuracyls))*100,2)
        hist_avg['time'] = sum(timels) / len(timels)
        hist_avg['param'] = param
        hist_avg['model_type'] = "RBM"
        HelperMethods.save_history(hist_avg)
    else:
        train_data_x, train_data_y,test_data_x,test_data_y,validation_data_x,validation_data_y, classes = DataPreprocessing.get_digits_dataset(0.1,True,model_type='RBM')
        history, time, train_accuracy_original,  test_accuracy_original, train_accuracy ,  test_accuracy = Run_RBM.run(train_data_x,train_data_y,test_data_x,test_data_y,validation_data_x,validation_data_y,10,n_hin=hidden_layers)

        HelperMethods.plot_results(history,train_accuracy,test_accuracy,time,"No Cross-Validation Results")
        print("Average epoch time: "+str(sum(history['epoch_time'])/len(history['epoch_time'])))
        HelperMethods.plot_result_single(history['epoch_time'], "Epoch Time Plot", "Epochs", "Time Elapsed")

        history['train_accuracy'] = train_accuracy
        history['test_accuracy'] =  test_accuracy
        history['train_accuracy_original'] = train_accuracy_original
        history['test_accuracy_original'] =  test_accuracy_original
        history['time'] = time
        history['param'] = param
        history['model_type'] = "RBM"
        HelperMethods.save_history(history)
else:
    train_data_x, train_data_y, test_data_x, test_data_y,validation_data_x,validation_data_y, classes = DataPreprocessing.get_digits_dataset(0.2, True,model_type='RBM')
    Run_RBM.run(train_data_x, train_data_y, test_data_x, test_data_y,validation_data_x,validation_data_y,num_classes=classes,mode=mode,n_hin=hidden_layers)
