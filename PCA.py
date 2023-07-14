import time
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import DataPreprocessing
from sklearn.manifold import TSNE

train_data_x, train_data_y, test_data_x, test_data_y, validation_data_x, validation_data_y, classes = DataPreprocessing.get_digits_dataset(0.1, True, model_type='AE')

start = time.time()
pca = PCA(n_components=500)
pca_features_train = pca.fit_transform(train_data_x)
pca_features_test = pca.fit_transform(test_data_x)
end = time.time()

clf = LinearDiscriminantAnalysis()
clf.fit(train_data_x, train_data_y)
predictions = clf.predict(test_data_x)
pred_accuracy_pca_test = sum(predictions == test_data_y) / test_data_y.shape[0]
predictions = clf.predict(train_data_x)
pred_accuracy_pca_train = sum(predictions == train_data_y) / train_data_y.shape[0]

print("Train accuracy LDA: "+str(round(pred_accuracy_pca_train*100,2))+"%")
print("Train accuracy LDA: "+str(round(pred_accuracy_pca_test*100,2))+"%")
print("---------------------")

clf = LinearDiscriminantAnalysis()
clf.fit(pca_features_train, train_data_y)
predictions = clf.predict(pca_features_test)
pred_accuracy_pca_test = sum(predictions == test_data_y) / test_data_y.shape[0]
predictions = clf.predict(pca_features_train)
pred_accuracy_pca_train = sum(predictions == train_data_y) / train_data_y.shape[0]

print("Train accuracy PCA+LDA: "+str(round(pred_accuracy_pca_train*100,2))+"%")
print("Train accuracy PCA+LDA: "+str(round(pred_accuracy_pca_test*100,2))+"%")
print("PCA time: "+str(end-start))

import seaborn as sns

pca = TSNE(n_components=2)
principalComponents = pca.fit_transform(test_data_x)
df = pd.DataFrame(principalComponents, columns=['Feature 1', 'Feature 2'])
df['Class'] = test_data_y
sns.pairplot(df, hue='Class')
plt.show()