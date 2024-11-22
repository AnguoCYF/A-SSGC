import pandas as pd
from sklearn.preprocessing import StandardScaler

from A_SSGC import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Data Preprocessing
df = pd.read_csv('UCI_Faults.csv')
# df = pd.read_csv('Kaggle_Faults.csv')
features = df.iloc[:, :-7].values
labels = df.iloc[:, -7:].values
labels = np.argmax(labels, axis=1)
num_classes = len(np.unique(labels))

scaler = StandardScaler()
features = scaler.fit_transform(features)

train_ratio = 0.8
random_state = 42

features, labels, train_mask, test_mask = generate_masks(features, labels, train_ratio, seed=random_state,
                                                         resample=True)
#%%
X_train = features[train_mask.cpu().numpy()]
X_test = features[test_mask.cpu().numpy()]

y_train = labels[train_mask.cpu().numpy()]
y_test = labels[test_mask.cpu().numpy()]

#%%
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Use Support Vector Machine (SVM) for classification
model_svm = SVC(random_state=random_state).fit(X_train, y_train)
y_preds_svm = model_svm.predict(X_test)

prec_svm, rec_svm, f1_svm, num_svm = precision_recall_fscore_support(y_test, y_preds_svm, average='macro')
acc_svm = accuracy_score(y_test, y_preds_svm)

print("SVM Classifier")
print("Macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_svm, rec_svm, f1_svm, acc_svm))


#%%
from sklearn.neighbors import KNeighborsClassifier

# Use k-Nearest Neighbors (k-NN) for classification
model_knn = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
y_preds_knn = model_knn.predict(X_test)

prec_knn, rec_knn, f1_knn, num_knn = precision_recall_fscore_support(y_test, y_preds_knn, average='macro')
acc_knn = accuracy_score(y_test, y_preds_knn)

print("\nk-NN Classifier")
print("Macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_knn, rec_knn, f1_knn, acc_knn))

#%%
from sklearn.naive_bayes import GaussianNB

# Use Naive Bayes (NB) for classification
model_nb = GaussianNB().fit(X_train, y_train)
y_preds_nb = model_nb.predict(X_test)

prec_nb, rec_nb, f1_nb, num_nb = precision_recall_fscore_support(y_test, y_preds_nb, average='macro')
acc_nb = accuracy_score(y_test, y_preds_nb)

print("\nNaive Bayes Classifier")
print("Macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_nb, rec_nb, f1_nb, acc_nb))

#%%
# Use Logistic Regression for classification
model_lr = LogisticRegression(max_iter=500, random_state=random_state).fit(X_train, y_train)
y_preds_lr = model_lr.predict(X_test)

prec_lr, rec_lr, f1_lr, num_lr = precision_recall_fscore_support(y_test, y_preds_lr, average='macro')
acc_lr = accuracy_score(y_test, y_preds_lr)

print("Logistic Regression Classifier")
print("Macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_lr, rec_lr, f1_lr, acc_lr))

#%%
from sklearn.neural_network import MLPClassifier

out_dim = X_train.shape[1]
num_classes = len(set(y_train))

# Train the MLP Classifier
model_mlp = MLPClassifier(hidden_layer_sizes=(out_dim,), activation='relu', max_iter=500, random_state=random_state)
model_mlp.fit(X_train, y_train)

# Make predictions
y_preds_mlp = model_mlp.predict(X_test)

# Evaluate the MLP Classifier
prec_mlp, rec_mlp, f1_mlp, num_mlp = precision_recall_fscore_support(y_test, y_preds_mlp, average='macro')
acc_mlp = accuracy_score(y_test, y_preds_mlp)

print("MLP Classifier")
print("Macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_mlp, rec_mlp, f1_mlp, acc_mlp))

#%%
from sklearn.tree import DecisionTreeClassifier

# Use Decision Tree for classification
model_dt = DecisionTreeClassifier(random_state=random_state).fit(X_train, y_train)
y_preds_dt = model_dt.predict(X_test)

prec_dt, rec_dt, f1_dt, num_dt = precision_recall_fscore_support(y_test, y_preds_dt, average='macro')
acc_dt = accuracy_score(y_test, y_preds_dt)

print("Decision Tree Classifier")
print("Macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_dt, rec_dt, f1_dt, acc_dt))
