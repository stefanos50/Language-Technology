import HelperMethods

saved = HelperMethods.retrieve_history()
losses = []
accuraces = []
val_losses = []
val_accuraces = []
epoch_times = []
params = []
for save in saved:
  losses.append(saved[save]['loss'])
  accuraces.append(saved[save]['accuracy'])
  val_accuraces.append(saved[save]['val_accuracy'])
  val_losses.append(saved[save]['val_loss'])
  epoch_times.append(saved[save]['epoch_time'])
  params.append(saved[save]['param'])

  print("------")
  print(str(saved[save]['param'])+" train accuracy: "+str(saved[save]['train_accuracy'])+"%")
  print(str(saved[save]['param']) + " test accuracy: " + str(saved[save]['test_accuracy'])+"%")
  print(str(saved[save]['param']) + " time: " + str(saved[save]['time'])+"%")

  if(saved[save]['model_type']=='RBM'):
    print(str(saved[save]['param']) + " train original accuracy: " + str(saved[save]['train_accuracy_original']) + "%")
    print(str(saved[save]['param']) + " test original accuracy: " + str(saved[save]['test_accuracy_original']) + "%")
  print("------")

HelperMethods.plot_result_multiple(losses,"Loss Comparison","Epoch","Loss",params)
HelperMethods.plot_result_multiple(accuraces,"Accuracy Comparison","Epoch","Loss",params)
HelperMethods.plot_result_multiple(val_accuraces,"Val Accuracy Comparison","Epoch","Loss",params)
HelperMethods.plot_result_multiple(val_losses,"Val Loss Comparison","Epoch","Loss",params)
HelperMethods.plot_result_multiple(epoch_times,"Epoch Time Comparison","Epoch","Loss",params)
