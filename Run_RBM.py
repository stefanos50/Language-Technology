import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import HelperMethods
from DBN import DBN


def run(X_train,y_train,X_test,y_test,X_val,y_val,num_classes,val_size=0.1,batch_size=32,mode="train",n_hin=None,classification=True,device_name="cuda"):

    device = HelperMethods.initialize_hardware(device_name)

    num_features = X_train.shape[1]
    arch = [num_features] + n_hin

    if len(arch)>2:
        algo_used = 'DBN'
    else:
        algo_used = 'RBM'

    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=1)

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)

    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    X_val = torch.Tensor(X_val)
    y_val = torch.Tensor(y_val)

    Traindataset = TensorDataset(X_train, y_train)
    Valdataset = TensorDataset(X_val, y_val)
    Testdataset = TensorDataset(X_test, y_test)

    trainDataLoader = DataLoader(Traindataset, shuffle=True, batch_size=batch_size, drop_last=True)
    valDataLoader = DataLoader(Valdataset, shuffle=True, batch_size=batch_size, drop_last=True)
    testDataLoader = DataLoader(Testdataset, batch_size=batch_size)

    if mode == "train":

        #model = RBM(n_vis=num_features,n_hin=n_hin,k=5,device=device,optimizer_name='RMSprop',learning_rate= 0.0001,weight_decay=0.0001,momentum=1,weight_initializer='xavier_uniform').to(device)
        model = DBN(num_classes=num_classes,hidden_units=arch,device=device).to(device)

        start = time.time()
        model.fit(num_epochs=10,tuning_epochs=30,train_loader=trainDataLoader,val_loader=valDataLoader,tuning=(len(arch)>2),classification=classification)
        end = time.time()

        train_acc_original,test_acc_original,train_acc,test_acc = model.evaluate(trainDataLoader,testDataLoader,batch_size,num_features,n_hin[-1])
        print('Train accuracy of original data: '+str(round(train_acc_original*100,2))+"%")
        print('Test accuracy of original data: '+str(round(test_acc_original*100,2))+"%")
        print('Train accuracy of '+algo_used+' features using LDA: '+str(round(train_acc*100,2))+"%")
        print('Test accuracy of '+algo_used+' features using LDA: '+str(round(test_acc*100,2))+"%")

        if len(arch) > 2 and classification:
            train_acc = model.evaluate_dbn(trainDataLoader,True).cpu().detach().numpy().tolist()
            test_acc = model.evaluate_dbn(testDataLoader,True).cpu().detach().numpy().tolist()

            print('Train accuracy: ' + str(round(train_acc * 100,2)) + "%")
            print('Test accuracy: ' + str(round(test_acc * 100,2)) + "%")


        return model.History, end - start, round(train_acc_original*100,2),  round(test_acc_original*100,2), round(train_acc*100,2) ,  round(test_acc*100,2)
    else:

        #model = RBM(n_vis=num_features,n_hin=n_hin,k=5,device=device,optimizer_name='adam',learning_rate= 0.0001,weight_decay=1e-5,momentum=1).to(device)
        model = DBN(num_classes=num_classes,hidden_units=arch,device=device).to(device)
        model.load_state_dict(torch.load("saved_"+algo_used+"_model.pth",map_location=torch.device(device_name)))
        model.eval()

        train_acc_original,test_acc_original,train_acc,test_acc = model.evaluate(trainDataLoader,testDataLoader,batch_size,num_features,n_hin[-1])

        print('Train accuracy of original data: '+str(round(train_acc_original*100,2))+"%")
        print('Test accuracy of original data: '+str(round(test_acc_original*100,2))+"%")
        print('Train accuracy of '+algo_used+' features using LDA: '+str(round(train_acc*100,2))+"%")
        print('Test accuracy of '+algo_used+' features using LDA: '+str(round(test_acc*100,2))+"%")

        if len(arch) > 2 and classification:
            train_acc = model.evaluate_dbn(trainDataLoader,True).cpu().detach().numpy().tolist()
            test_acc = model.evaluate_dbn(testDataLoader,True).cpu().detach().numpy().tolist()

            print('Train accuracy: ' + str(round(train_acc * 100,2)) + "%")
            print('Test accuracy: ' + str(round(test_acc * 100,2)) + "%")

            real_names = ['0','1','2','3','4','5','6','7','8','9']
            for i in range(num_classes):
                data_class = []
                labels_class = []
                for idx, (data, target) in enumerate(testDataLoader):
                    for j in range(len(data)):
                        if np.argmax(target[j].numpy()) == i:
                            data_class.append(data[j].numpy().tolist())
                            labels_class.append(target[j].numpy().tolist())
                data_class = torch.Tensor(np.array(data_class))
                labels_class = torch.Tensor(np.array(labels_class))
                classdataset = TensorDataset(data_class, labels_class)
                classDataLoader = DataLoader(classdataset, batch_size=batch_size)
                print("Model Accuracy for class "+str(real_names[i])+": " + str(round(model.evaluate_dbn(classDataLoader).cpu().detach().numpy().tolist() * 100, 2)) + "%")


