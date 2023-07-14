import time
import librosa
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.nn import Module
from torch.nn import Linear
from torch import nn
from torch.optim import Adam,SGD,RMSprop,Adagrad,Adadelta,AdamW,Adamax
from torchshape import tensorshape
import seaborn as sns
from EarlyStopper import EarlyStopper

class FNetwork(Module):
	def init_loss_function(self,function_name):
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

	def init_optimizer(self,optimizer_name=None,learning_rate=None,momentum=None,weight_decay=None):
		if optimizer_name=="adam":
			self.opt = Adam(self.parameters(), lr=learning_rate,weight_decay=weight_decay)
		elif optimizer_name=="SGD":
			self.opt = SGD(self.parameters(), lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
		elif optimizer_name=="RMSprop":
			self.opt = RMSprop(self.parameters(), lr=learning_rate,weight_decay=weight_decay)
		elif optimizer_name=="Adagrad":
			self.opt = Adagrad(self.parameters(), lr=learning_rate,weight_decay=weight_decay)
		elif optimizer_name=="Adadelta":
			self.opt = Adadelta(self.parameters(), lr=learning_rate,weight_decay=weight_decay)
		elif optimizer_name=="AdamW":
			self.opt = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
		elif optimizer_name=="Adamax":
			self.opt = Adamax(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

	def init_weights(self,m,init_method,gain):
		if init_method == "xavier_uniform":
			torch.nn.init.xavier_uniform_(m.weight,gain)
		elif init_method == "xavier_normal":
			torch.nn.init.xavier_normal_(m.weight,gain)
		elif init_method == "kaiming_normal":
			torch.nn.init.kaiming_normal(m.weight)
		elif init_method == "kaiming_uniform":
			torch.nn.init.kaiming_uniform(m.weight)
		elif init_method == "orthogonal":
			torch.nn.init.orthogonal(m.weight)


	def __init__(self, classes=1,loss_function="cross-entropy",optimizer="adam",learning_rate=0.001,input_shape=None,device=None,architecture=None,early_stop=True,waiting=555,min_delta=0,momentum=1,weight_decay=1e-5,model_type=None,weight_initializer="xavier_uniform"):
		# call the parent constructor
		super(FNetwork, self).__init__()

		layers = []
		last_output_shape = 0
		for layer_name, params in architecture.items():
			if 'activation' in layer_name:
				layers.append(params['function'])
			elif 'Linear' in layer_name:
				if len(layers) == 0:
					layers.append(Linear(in_features=input_shape[1], out_features=params['out_features']))
					last_output_shape = tensorshape(layers[-1], input_shape)
				elif layer_name == list(architecture.keys())[-2]:
					layers.append(Linear(in_features=last_output_shape[1], out_features=classes))
				else:
					if layers[0] == 'Decoder':
						layers = []
					layers.append(Linear(in_features=last_output_shape[1], out_features=params['out_features']))
					last_output_shape = tensorshape(layers[-1], last_output_shape)
			elif 'Dropout' in layer_name:
				layers.append(params['function'])
				last_output_shape = tensorshape(layers[-1], last_output_shape)
			elif 'BatchNorm' in layer_name:
				if 'BatchNorm1d' in layer_name:
					layers.append(nn.BatchNorm1d(last_output_shape[1]))
					last_output_shape = tensorshape(layers[-1], last_output_shape)
				elif 'BatchNorm2d' in layer_name:
					layers.append(nn.BatchNorm2d(last_output_shape[1]))
					last_output_shape = tensorshape(layers[-1], last_output_shape)
				else:
					print('Wrong batch normalization definition...')
					exit(1)
			elif 'Decoder' in layer_name:
				self.encoder_layers = nn.ModuleList(layers)
				layers = ['Decoder']
		self.model_type = model_type
		if self.model_type == 'FN':
			self.arch_layers = nn.ModuleList(layers)
		else:
			self.decoder_layers = nn.ModuleList(layers)

		if self.model_type == 'FN':
			for m in self.arch_layers:
				if isinstance(m, nn.Linear):
					self.init_weights(m,weight_initializer,0.5)
		else:
			for m in self.encoder_layers:
				if isinstance(m, nn.Linear):
					self.init_weights(m,weight_initializer,0.5)
			for m in self.decoder_layers:
				if isinstance(m, nn.Linear):
					self.init_weights(m,weight_initializer,0.5)

		self.init_loss_function(loss_function)
		self.init_optimizer(optimizer_name=optimizer,learning_rate=learning_rate,momentum=momentum,weight_decay=weight_decay)
		self.verbose_levels = [0, 1, 10, 100, 1000]
		self.History = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "epoch_time": []}
		self.device = device
		self.early_stop = early_stop
		if self.early_stop:
			self.early_stopper = EarlyStopper(waiting=waiting, mind=min_delta)

	def save_model(self):
		torch.save(self.state_dict(), "saved_"+str(self.model_type)+"_model.pth")

	def forward(self, x , decode=True):
		if self.model_type == 'FN':
			for _, layer in enumerate(self.arch_layers, start=0):
				x = layer(x)
			return x
		else:
			if decode:
				for _, layer in enumerate(self.encoder_layers, start=0):
					x = layer(x)
				for _, layer in enumerate(self.decoder_layers, start=0):
					x = layer(x)
				return x
			else:
				for _, layer in enumerate(self.encoder_layers, start=0):
					x = layer(x)
				return x

	def evaluate(self,xDataLoader,plot_conf=False):
		accuracy = 0
		predicted_list = []
		original_list = []
		with torch.no_grad():
			self.eval()
			for (x, y) in xDataLoader:
				(x, y) = (x.to(self.device), y.to(self.device))
				predicted = self(x)
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

		return accuracy/len(xDataLoader)

	def evaluate_ae(self, train_loader, test_loader, batch_size, input_size, output_size, image_size=(40, 74)):

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
				(batch, labels) = (batch.to(self.device), labels.to(self.device))
				batch = Variable(batch.view(-1, input_size))  # flatten input data

				train_features[i * batch_size:i * batch_size + len(batch)] = self(batch,False).cpu().data.numpy()
				train_labels[i * batch_size:i * batch_size + len(batch)] = labels.cpu().data.numpy()

				train_features_original[i * batch_size:i * batch_size + len(batch)] = batch.cpu().data.numpy()
				train_labels_original[i * batch_size:i * batch_size + len(batch)] = labels.cpu().data.numpy()

			for i, (batch, labels) in enumerate(test_loader):
				(batch, labels) = (batch.to(self.device), labels.to(self.device))
				batch = Variable(batch.view(-1, input_size))  # flatten input data

				test_features[i * batch_size:i * batch_size + len(batch)] = self(batch,False).cpu().data.numpy()
				test_labels[i * batch_size:i * batch_size + len(batch)] = labels.cpu().data.numpy()

				test_features_original[i * batch_size:i * batch_size + len(batch)] = batch.cpu().data.numpy()
				test_labels_original[i * batch_size:i * batch_size + len(batch)] = labels.cpu().data.numpy()

		clf = LinearDiscriminantAnalysis()
		clf.fit(train_features, train_labels)
		predictions = clf.predict(test_features)
		cm = confusion_matrix(test_labels, predictions,normalize='pred')
		pred_accuracy_ae_test = sum(predictions == test_labels) / test_labels.shape[0]
		predictions = clf.predict(train_features)
		pred_accuracy_ae_train = sum(predictions == train_labels) / train_labels.shape[0]

		clf = LinearDiscriminantAnalysis()
		clf.fit(train_features_original, train_labels_original)
		predictions = clf.predict(test_features_original)
		pred_accuracy_ae_test_original = sum(predictions == test_labels_original) / test_labels_original.shape[0]
		predictions = clf.predict(train_features_original)
		pred_accuracy_ae_train_original = sum(predictions == train_labels_original) / train_labels_original.shape[0]

		print("Train Set Original Features Accuracy: "+str(round(pred_accuracy_ae_train_original*100,2))+"%")
		print("Test Set Original Features Accuracy: "+str(round(pred_accuracy_ae_test_original*100,2))+"%")
		print("Train Set Autoencoder Features Accuracy: "+str(round(pred_accuracy_ae_train*100,2))+"%")
		print("Test Set Autoencoder Features Accuracy: "+str(round(pred_accuracy_ae_test*100,2))+"%")

		tsne = TSNE(n_components=2)
		tsneComponents = tsne.fit_transform(test_features)
		df = pd.DataFrame(tsneComponents, columns=['Feature 1', 'Feature 2'])
		df['Class'] = test_labels
		sns.pairplot(df, hue='Class')
		plt.show()

		fig, ax = plt.subplots(figsize=(8, 6))
		sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False)
		ax.set_xlabel('Predicted labels')
		ax.set_ylabel('True labels')
		ax.set_title('Confusion Matrix')
		ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
		ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
		plt.show()

		return pred_accuracy_ae_train_original, pred_accuracy_ae_test_original, pred_accuracy_ae_train, pred_accuracy_ae_test

	def plot_mel_spectrogram(self,S_db,S_db2):
		fig, ax = plt.subplots(1,2)
		img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax[0])
		ax[0].set(title='Original')
		fig.colorbar(img, ax=ax[0], format="%+2.f dB")

		img2 = librosa.display.specshow(S_db2, x_axis='time', y_axis='linear', ax=ax[1])
		ax[1].set(title='Decoded')
		fig.colorbar(img2, ax=ax[1], format="%+2.f dB")
		plt.show()

	def compare_images(self,imagesloader,img_shape,img_to_show=2):
		with torch.no_grad():
			self.eval()
			for (x, y) in imagesloader:
				if img_to_show == 0:
					break
				(x, y) = (x.to(self.device), y.to(self.device))
				predicted = self(x)
				for i in range(x.shape[0]):
					if img_to_show == 0:
						break
					self.plot_mel_spectrogram(np.reshape(x[i].cpu().data.numpy(), img_shape),np.reshape(predicted[i].cpu().data.numpy(),img_shape))

	def calculate_accuracy(self,pred,true):
		return torch.sum(pred.argmax(1) == true.argmax(1))/len(true)

	def print_progress(self, phase, accuracy, loss, current_epoch,verbose):
		if current_epoch % self.verbose_levels[verbose] == 0:
			print("Phase " + str(phase) + " - Accuracy: " + str(accuracy * 100) + "%" + " - " + "loss: " + str(loss))

	def evaluate_autoencoder(self,xDataLoader):
		total_loss = 0
		with torch.no_grad():
			self.eval()
			for (x, y) in xDataLoader:
				(x, y) = (x.to(self.device), y.to(self.device))
				reconstructed = self(x)
				loss = self.loss_fn(reconstructed, x)
				total_loss += loss
		return loss / len(xDataLoader)

	def fit(self,num_epochs=10,trainDataLoader=None,valDataLoader=None,verbose=1):
		lossFn = self.loss_fn

		# loop over our epochs
		for epoch in range(0, num_epochs):
			start = time.time()
			if (verbose != 0):
				if (epoch+1) % self.verbose_levels[verbose] == 0:
					print("\n")
					print("Epoch: " + str(epoch+1) + "/" + str(num_epochs) + " - ║{0:20s}║ {1:.1f}%".format(
						'🟩' * int((epoch+1) / num_epochs * 20), (epoch+1) / num_epochs * 100))
			# set the model in training mode
			self.train()
			total_train_loss = 0
			total_val_loss = 0
			total_train_accuracy = 0
			total_val_accuracy = 0
			# loop over the training set
			for (x, y) in trainDataLoader:
				(x, y) = (x.to(self.device), y.to(self.device))
				predicted = self(x)

				if self.model_type == 'AE':
					loss = lossFn(predicted, x)
				else:
					loss = lossFn(predicted, y)

				self.opt.zero_grad()
				loss.backward()
				self.opt.step()

				total_train_loss += loss
				if self.model_type == 'FN':
					accuracy = self.calculate_accuracy(predicted,y)
				else:
					accuracy = 0
				total_train_accuracy += accuracy
			if self.model_type == 'FN':
				self.print_progress("Train",(total_train_accuracy/len(trainDataLoader)).item(),(total_train_loss/len(trainDataLoader)).item(),epoch+1,verbose)
			else:
				self.print_progress("Train",0,(total_train_loss / len(trainDataLoader)).item(), epoch + 1, verbose)

			with torch.no_grad():
				self.eval()
				for (x, y) in valDataLoader:
					(x, y) = (x.to(self.device), y.to(self.device))
					predicted = self(x)
					if self.model_type == 'AE':
						loss = lossFn(predicted, x)
					else:
						loss = lossFn(predicted, y)
					total_val_loss += loss
					if self.model_type == 'FN':
						accuracy = self.calculate_accuracy(predicted,y)
					else:
						accuracy = 0
					total_val_accuracy += accuracy
				if self.model_type == 'FN':
					self.print_progress("Val", (total_val_accuracy/len(valDataLoader)).item(),(total_val_loss/len(valDataLoader)).item(), epoch + 1,verbose)
				else:
					self.print_progress("Val", 0,(total_val_loss / len(valDataLoader)).item(), epoch + 1, verbose)
			end = time.time()
			print("Epoch time elapsed: "+str((end-start)))
			self.History["loss"].append((total_train_loss.cpu().detach().numpy()/len(trainDataLoader)))
			if self.model_type == 'FN':
				self.History["accuracy"].append((total_train_accuracy.cpu().detach().numpy()/len(trainDataLoader)))
			else:
				self.History["accuracy"].append(0)
			self.History["val_loss"].append((total_val_loss.cpu().detach().numpy()/len(valDataLoader)))
			if self.model_type == 'FN':
				self.History["val_accuracy"].append((total_val_accuracy.cpu().detach().numpy()/len(valDataLoader)))
			else:
				self.History["val_accuracy"].append(0)
			self.History["epoch_time"].append(end-start)

			if self.early_stop:
				if self.early_stopper.early_stop((total_val_loss/len(valDataLoader)).item()):
					break
		self.save_model()








