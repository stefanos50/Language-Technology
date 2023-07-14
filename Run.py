import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import device
from torch.utils.data import TensorDataset, DataLoader
import HelperMethods
from FNetwork import FNetwork


def run(X_train,y_train,X_test,y_test,X_val,y_val,num_classes,val_size=0.1,batch_size=32,arch=None,mode="train",model_type=None,device_name='cuda'):

    device = HelperMethods.initialize_hardware(device_name)

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

    trainDataLoader = DataLoader(Traindataset, shuffle=True, batch_size=batch_size)
    valDataLoader = DataLoader(Valdataset, shuffle=True, batch_size=batch_size)
    testDataLoader = DataLoader(Testdataset, batch_size=batch_size)

    if mode == "train":

        if model_type == 'FN':
            model = FNetwork(classes=num_classes,learning_rate=0.001,optimizer="adam", loss_function='cross-entropy', device=device, input_shape=X_train.shape, architecture=arch,weight_decay=1e-07,momentum=1,model_type=model_type,weight_initializer="xavier_uniform").to(device)

        else:
            model = FNetwork(classes=X_train.shape[1],learning_rate=0.0001,optimizer="adam", loss_function='L1', device=device, input_shape=X_train.shape, architecture=arch,weight_decay=0.0001,momentum=1,model_type=model_type,waiting=5,weight_initializer="xavier_uniform").to(device)

        start = time.time()
        model.fit(num_epochs=100, trainDataLoader=trainDataLoader,valDataLoader= valDataLoader)
        end = time.time()

        if model_type == 'FN':
            accuracy_train = round(model.evaluate(trainDataLoader,True).cpu().detach().numpy().tolist() * 100, 2)
            accuracy_test = round(model.evaluate(testDataLoader,True).cpu().detach().numpy().tolist() * 100, 2)

        if model_type == 'AE':
            model.evaluate_ae(trainDataLoader,testDataLoader,batch_size,960,500)
            accuracy_train=model.evaluate_autoencoder(trainDataLoader).cpu().detach().numpy().tolist()
            accuracy_test=model.evaluate_autoencoder(testDataLoader).cpu().detach().numpy().tolist()
            print("Reconstructed train average loss: "+str(accuracy_train))
            print("Reconstructed test average loss: " + str(accuracy_test))
            #model.compare_images(testDataLoader,(20,48))

        return model.History, end - start, accuracy_train, accuracy_test
    else:
        X_train = torch.Tensor(X_train)
        if model_type == 'FN':
            model = FNetwork(classes=num_classes, loss_function='cross-entropy', device=device,
                             input_shape=X_train.shape, architecture=arch, learning_rate=0.001, optimizer="adam",
                             momentum=0.7, model_type=model_type).to(device)
            model.load_state_dict(torch.load("saved_"+str(model_type)+"_model.pth",map_location=torch.device(device_name)))
            model.eval()
            print("Train accuracy of the saved model: "+str(round(model.evaluate(trainDataLoader,True).cpu().detach().numpy().tolist() * 100, 2))+"%")
            print("Test accuracy of the saved model: "+str(round(model.evaluate(testDataLoader,True).cpu().detach().numpy().tolist() * 100, 2))+"%")

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
                print("Model Accuracy for class "+str(real_names[i])+": " + str(round(model.evaluate(classDataLoader).cpu().detach().numpy().tolist() * 100, 2)) + "%")
        else:
            model = FNetwork(classes=X_train.shape[1],learning_rate=0.0001,optimizer="AdamW", loss_function='L1', device=device, input_shape=X_train.shape, architecture=arch,weight_decay=0.0001,momentum=1,model_type=model_type).to(device)
            model.load_state_dict(torch.load("saved_"+str(model_type)+"_model.pth",map_location=torch.device(device_name)))
            model.eval()
            model.evaluate_ae(trainDataLoader,testDataLoader,batch_size,960,500)
            accuracy_train=model.evaluate_autoencoder(trainDataLoader).cpu().detach().numpy().tolist()
            accuracy_test=model.evaluate_autoencoder(testDataLoader).cpu().detach().numpy().tolist()
            print("Reconstructed train average loss: "+str(accuracy_train))
            print("Reconstructed test average loss: " + str(accuracy_test))
            #model.compare_images(testDataLoader,(20,48))