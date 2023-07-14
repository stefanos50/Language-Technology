import time
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.optim import Adam, SGD, RMSprop, Adagrad, Adadelta, AdamW
from EarlyStopper import EarlyStopper
from RBM import RBM
from torch.autograd import Variable
import seaborn as sns

class DBN(nn.Module):
    def init_optimizer(self, optimizer_name=None, learning_rate=None, momentum=None, weight_decay=None):
        if optimizer_name == "adam":
            self.opt = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            self.opt = SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "RMSprop":
            self.opt = RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "Adagrad":
            self.opt = Adagrad(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "Adadelta":
            self.opt = Adadelta(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def init_loss_function(self, function_name):
        if function_name == "cross-entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        elif function_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif function_name == "L1":
            self.loss_fn = nn.L1Loss()
        elif function_name == "binary-cross-entropy":
            self.loss_fn = nn.BCELoss()
        elif function_name == "neg-log-likelihood":
            self.loss_fn = nn.NLLLoss()

    def init_weights(self, m, init_method, gain):
        if init_method == "xavier_uniform":
            torch.nn.init.xavier_uniform_(m.weight, gain)
        elif init_method == "xavier_normal":
            torch.nn.init.xavier_normal_(m.weight, gain)
        elif init_method == "kaiming_normal":
            torch.nn.init.kaiming_normal(m.weight)
        elif init_method == "kaiming_uniform":
            torch.nn.init.kaiming_uniform(m.weight)
        elif init_method == "orthogonal":
            torch.nn.init.orthogonal(m.weight)

    def __init__(self,num_classes, hidden_units , device , waiting=10,min_delta=0,early_stop=False,optimizer_name="adam",learning_rate=0.0001,momentum=0.6,weight_decay=1e-7,loss_function="cross-entropy",weight_initializer="xavier_uniform",gain=1.0):
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList([
            #RBM(n_vis=hidden_units[i - 1], n_hin=hidden_units[i], k=5, device=device, optimizer_name='RMSprop',learning_rate=0.0001, weight_decay=0.0001, momentum=1, weight_initializer='xavier_uniform') for i in range(1, len(hidden_units))
            RBM(n_vis=hidden_units[i-1],n_hin=hidden_units[i],k=7,device=device,optimizer_name='SGD',learning_rate= 0.0001,weight_decay=1e-5,momentum=0.6,weight_initializer='xavier_uniform') for i in range(1,len(hidden_units))
        ])

        self.nnet =  nn.ModuleList([
            nn.Linear(hidden_units[-1], num_classes),
            nn.LogSoftmax(dim=1)
        ])

        for m in self.nnet:
            if isinstance(m, nn.Linear):
                self.init_weights(m,weight_initializer,gain)

        self.init_optimizer(optimizer_name,learning_rate,momentum,weight_decay)
        self.init_loss_function(loss_function)
        self.History = {}
        self.device = device
        self.verbose_levels = [0, 1, 10, 100, 1000]

        self.early_stop = early_stop
        if self.early_stop:
            self.early_stopper = EarlyStopper(waiting=waiting, mind=min_delta)

    def forward(self, v , out=False):
        h = v
        for rbm in self.rbms:
            h = rbm.v_to_h(h)[0]
        if out == True:
            for layer in self.nnet:
                h = layer(h)
        return h

    def print_progress(self, phase, accuracy, loss, current_epoch, verbose):
        if current_epoch % self.verbose_levels[verbose] == 0:
            print("Phase " + str(phase) + " - Accuracy: " + str(accuracy * 100) + "%" + " - " + "loss: " + str(loss))

    def calculate_accuracy(self, pred, true):
        return torch.sum(pred.argmax(1) == true.argmax(1)) / len(true)

    def evaluate_dbn(self, xDataLoader, plot_conf=False):
        accuracy = 0
        predicted_list = []
        original_list = []
        with torch.no_grad():
            self.eval()
            for (x, y) in xDataLoader:
                (x, y) = (x.to(self.device), y.to(self.device))
                predicted = self(x,True)
                accuracy += self.calculate_accuracy(predicted, y)
                predicted_list.append(predicted.cpu().detach())
                original_list.append(y.cpu().detach())

        if plot_conf:
            concatenated_tensor = torch.cat(predicted_list, dim=0)
            predicted_list = concatenated_tensor.numpy()

            concatenated_tensor = torch.cat(original_list, dim=0)
            original_list = concatenated_tensor.numpy()

            cm = confusion_matrix(original_list.argmax(1), predicted_list.argmax(1),normalize='pred')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            plt.show()

        return accuracy / len(xDataLoader)

    def save_model(self,algo_name = "RBM"):
        torch.save(self.state_dict(), "saved_"+algo_name+"_model.pth")

    def evaluate(self,train_loader,test_loader,batch_size,input_size,output_size):

        train_features = np.zeros((len(train_loader) * batch_size, output_size))
        train_labels = np.zeros(len(train_loader) * batch_size)
        test_features = np.zeros((len(test_loader) * batch_size, output_size))
        test_labels = np.zeros(len(test_loader) * batch_size)

        train_features_original = np.zeros((len(train_loader) * batch_size, input_size))
        train_labels_original = np.zeros(len(train_loader) * batch_size)
        test_features_original = np.zeros((len(test_loader) * batch_size, input_size))
        test_labels_original = np.zeros(len(test_loader) * batch_size)

        with torch.no_grad():
            self.eval()
            for i, (batch, labels) in enumerate(train_loader):
                batch = Variable(batch.view(-1, input_size))  # flatten input data

                train_features[i * batch_size:i * batch_size + len(batch)] = self.forward(batch,out=False).cpu().data.numpy()
                train_labels[i * batch_size:i * batch_size + len(batch)] = labels.numpy().argmax(1)

                train_features_original[i * batch_size:i * batch_size + len(batch)] = batch.cpu().data.numpy()
                train_labels_original[i * batch_size:i * batch_size + len(batch)] = labels.numpy().argmax(1)

            for i, (batch, labels) in enumerate(test_loader):
                batch = Variable(batch.view(-1, input_size))  # flatten input data

                test_features[i * batch_size:i * batch_size + len(batch)] = self.forward(batch,out=False).cpu().data.numpy()
                test_labels[i * batch_size:i * batch_size + len(batch)] = labels.numpy().argmax(1)

                test_features_original[i * batch_size:i * batch_size + len(batch)] = batch.cpu().data.numpy()
                test_labels_original[i * batch_size:i * batch_size + len(batch)] = labels.numpy().argmax(1)

        clf = LinearDiscriminantAnalysis()
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        cm = confusion_matrix(test_labels, predictions,normalize='pred')
        pred_accuracy_rbm_test = sum(predictions == test_labels)/ test_labels.shape[0]
        predictions = clf.predict(train_features)
        pred_accuracy_rbm_train = sum(predictions == train_labels)/ train_labels.shape[0]


        clf = LinearDiscriminantAnalysis()
        clf.fit(train_features_original, train_labels_original)
        predictions = clf.predict(test_features_original)
        pred_accuracy_rbm_test_original = sum(predictions == test_labels_original)/ test_labels_original.shape[0]
        predictions = clf.predict(train_features_original)
        pred_accuracy_rbm_train_original = sum(predictions == train_labels_original)/ train_labels_original.shape[0]

        pca = TSNE(n_components=2)
        principalComponents = pca.fit_transform(test_features)
        df = pd.DataFrame(principalComponents, columns=['Feature 1', 'Feature 2'])
        df['Class'] = test_labels
        sns.pairplot(df,hue='Class')
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap="Blues", cbar=False)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['0', '1', '2','3','4','5','6','7','8','9'])
        ax.yaxis.set_ticklabels(['0', '1', '2','3','4','5','6','7','8','9'])
        plt.show()

        return pred_accuracy_rbm_train_original,pred_accuracy_rbm_test_original,pred_accuracy_rbm_train,pred_accuracy_rbm_test



    def fit(self,num_epochs=10,tuning_epochs=10,train_loader=None,val_loader=None,prob=0.5,verbose=1,tuning=False,classification=False):
        trainDataLoader = train_loader
        valDataLoader = val_loader
        for rbm in self.rbms:
            self.History = rbm.fit(num_epochs=num_epochs, train_loader=train_loader,val_loader= val_loader,prob=prob,verbose=verbose)

            xtrain = []
            ytrain = []
            xval = []
            yval = []
            for i, (batch, labels) in enumerate(train_loader):
                xtrain.append(rbm.v_to_h(batch)[0])
                ytrain.append(labels)
            for i, (batch, labels) in enumerate(val_loader):
                xval.append(rbm.v_to_h(batch)[0])
                yval.append(labels)

            xtrain = torch.stack(xtrain)
            ytrain = torch.stack(ytrain)
            xval = torch.stack(xval)
            yval = torch.stack(yval)

            train_loader = torch.utils.data.TensorDataset(xtrain, ytrain)
            train_loader = torch.utils.data.DataLoader(train_loader, batch_size=1)

            val_loader = torch.utils.data.TensorDataset(xval, yval)
            val_loader = torch.utils.data.DataLoader(val_loader, batch_size=1)

        if tuning == False:
            self.save_model("RBM")
            return
        if classification == False:
            self.save_model("DBN")
            return

        lossFn = self.loss_fn
        self.History = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "epoch_time": []}
        # loop over our epochs
        for epoch in range(0, tuning_epochs):

            start = time.time()
            if (verbose != 0):
                if (epoch + 1) % self.verbose_levels[verbose] == 0:
                    print("\n")
                    print("Epoch: " + str(epoch + 1) + "/" + str(tuning_epochs) + " - ║{0:20s}║ {1:.1f}%".format(
                        '🟩' * int((epoch + 1) / tuning_epochs * 20), (epoch + 1) / tuning_epochs * 100))
            # set the model in training mode
            self.train()
            total_train_loss = 0
            total_val_loss = 0
            total_train_accuracy = 0
            total_val_accuracy = 0
            # loop over the training set
            for (x, y) in trainDataLoader:
                (x, y) = (x.to(self.device), y.to(self.device))
                predicted = self.forward(x,out=True)

                loss = lossFn(predicted, y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_train_loss += loss

                accuracy = self.calculate_accuracy(predicted, y)

                total_train_accuracy += accuracy

            self.print_progress("Train", (total_train_accuracy / len(trainDataLoader)).item(),
                                    (total_train_loss / len(trainDataLoader)).item(), epoch + 1, verbose)

            with torch.no_grad():
                self.eval()
                for (x, y) in valDataLoader:
                    (x, y) = (x.to(self.device), y.to(self.device))
                    predicted = self.forward(x,out=True)

                    loss = lossFn(predicted, y)

                    total_val_loss += loss

                    accuracy = self.calculate_accuracy(predicted, y)

                    total_val_accuracy += accuracy

                self.print_progress("Val", (total_val_accuracy / len(valDataLoader)).item(),
                                        (total_val_loss / len(valDataLoader)).item(), epoch + 1, verbose)
            end = time.time()
            print("Epoch time elapsed: " + str((end - start)))
            self.History["loss"].append((total_train_loss.cpu().detach().numpy() / len(trainDataLoader)))

            self.History["accuracy"].append((total_train_accuracy.cpu().detach().numpy() / len(trainDataLoader)))

            self.History["val_loss"].append((total_val_loss.cpu().detach().numpy() / len(valDataLoader)))

            self.History["val_accuracy"].append((total_val_accuracy.cpu().detach().numpy() / len(valDataLoader)))

            self.History["epoch_time"].append(end - start)

            if self.early_stop:
                if self.early_stopper.early_stop((total_val_loss / len(valDataLoader)).item()):
                    break
        self.save_model("DBN")