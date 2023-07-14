import pickle
import torch
from matplotlib import pyplot as plt


def plot_result(x,y,title,y_label,x_label,x_legend,y_legend):
    plt.plot(x)
    plt.plot(y)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend([x_legend, y_legend], loc='upper left')
    plt.show()

def plot_result_multiple(x,title,y_label,x_label,params):
    for i in range(len(x)):
        plt.plot(x[i])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend([param for param in params], loc='upper left')
    plt.show()

def plot_result_single(x,title,y_label,x_label):
    plt.plot(x)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()

def plot_results(history,train_accuracy,test_accuracy,time,type):
    plot_result(history['loss'], history['val_loss'], "Loss Plot "+str(type), "Loss", "Epochs",
                              "Train Loss", "Val Loss")
    plot_result(history['accuracy'], history['val_accuracy'], "Accuracy Plot "+str(type), "Accuracy",
                              "Epochs", "Train Accuracy", "Val Accuracy")


    print("\n")
    print(str(type)+" train accuracy: " + str(train_accuracy) + "%")
    print(str(type)+" test accuracy: " + str(test_accuracy) + "%")
    print(str(type)+" fit execution time: " + str(time) + "s")

def history_average(history):
    loss = []
    val_loss = []
    accuracy = []
    val_accuracy = []
    epoch_time = []
    for dict in history:
        loss.append(dict['loss'])
        val_loss.append(dict['val_loss'])
        accuracy.append(dict['accuracy'])
        val_accuracy.append(dict['val_accuracy'])
        epoch_time.append(dict['epoch_time'])

    history_new = {}
    history_new['loss'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*loss)]
    history_new['val_loss'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*val_loss)]
    history_new['accuracy'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*accuracy)]
    history_new['val_accuracy'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*val_accuracy)]
    history_new['epoch_time'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*epoch_time)]
    return history_new

def initialize_hardware(hw_choice='cuda'):
    if hw_choice == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available() and hw_choice == 'cuda':
        print('Using device: ', device)
        print('Using gpu: ',torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        print('Using device: ', device)
    return device

def save_history(hist):
    try:
        prev_dict = retrieve_history()
    except:
        prev_dict = None
    if prev_dict == None:
        f = open("saved/history.pkl","wb")
        prev_dict = {}
        prev_dict[1] = hist
        pickle.dump(prev_dict, f)
        print('History saved successfully to file')
        f.close()
        return
    else:
        f = open("saved/history.pkl", "wb")
        prev_dict[list(prev_dict)[-1]+1] = hist
        pickle.dump(prev_dict, f)
        print('History saved successfully to file')
        f.close()
        return

def retrieve_history():
    # open a file, where you stored the pickled data
    file = open('saved/history.pkl', 'rb')

    # dump information to that file
    hist = pickle.load(file)

    # close the file
    file.close()

    return hist

