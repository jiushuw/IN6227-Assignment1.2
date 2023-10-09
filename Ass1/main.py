import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

df = pd.read_csv("D:/AY23-24/IN6227DataMining/Census Income Data Set/adult_total.csv")

columns_name = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status',
        'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
        'hours.per.week', 'native.country', 'income']

df_train = pd.read_csv("D:/AY23-24/IN6227DataMining/Census Income Data Set(1)/Census Income Data Set/adult.data", names=columns_name)
df_test = pd.read_csv("D:/AY23-24/IN6227DataMining/Census Income Data Set(1)/Census Income Data Set/adult.test", names=columns_name, skiprows=1)

# df_train.info()
# df_test.info()

# Missing value detection
df_train = df_train.replace(' ?', np.nan)
df_test = df_test.replace(' ?', np.nan)

df_train = df_train.dropna()
df_test = df_test.dropna()
print(df_train.isin([' ?']).sum())

print(df_train.describe(include=['O']))

# drop duplicates
df_train = df_train.drop_duplicates()
df_test = df_test.drop_duplicates()
'''
# **************************************
income_map = {'<=50K': 1, '>50K': 0}

df_train['income'] = df_train['income'].map(income_map)
df_train['income'] = df_train['income'].astype('int')


# basic information visualization
categorical = [col for col in df_train.columns if df_train[col].dtype == 'object' ]
numerical = [col for col in df_train.columns if df_train[col].dtype != 'object' ]

df_train[numerical].hist(bins=25, figsize=(7, 7))
plt.show()

num_rows = len(categorical)
num_cols = 3

fig = plt.figure(figsize=(12, 25))

for i, col in enumerate(categorical):
    ax = fig.add_subplot(num_rows, num_cols, i+1)
    df_train[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()
'''

# DATA ANALYSIS AND FEATURE ENGINEERING
# age
#sns.histplot(df_train, x='age', hue='income', bins= 32)
def age_group(df):
    age_bins = [0, 20, 40, 60, float('inf')]
    age_labels = ['1', '2', '3', '4']
    df_age_range = df.copy()
    df_age_range['age'] = pd.cut(df_age_range['age'], bins=age_bins, labels=age_labels)
    return df_age_range

df_train = age_group(df).copy()
df_test = age_group(df).copy()

#sns.histplot(df_train, x='age', hue='income', bins= 32)

age = pd.get_dummies(df_train['age'], drop_first=True)
df_train = df_train.drop('age', axis=1)
df_test = df_test.drop('age', axis=1)
df_train = pd.concat([df_train, age], axis=1)
df_test = pd.concat([df_test, age], axis=1)

# workclass

# sns.histplot(df_train, x='workclass', hue='income', bins= 32)
# plt.show()
df_train['workclass'] = df_train['workclass'].apply(lambda x: 1 if x == 'Private' else 0)
df_test['workclass'] = df_test['workclass'].apply(lambda x: 1 if x == 'Private' else 0)
# sns.countplot(data = df, x = 'workclass', hue = 'income')

df_train = df_train.drop('fnlwgt',axis=1)
df_test = df_test.drop('fnlwgt',axis=1)

# sns.countplot(data = df, x = 'education', hue = 'income')
# plt.tick_params(axis='x', rotation=90)

df_train = pd.get_dummies(df_train, columns=['education'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['education'], drop_first=True)
df_train = df_train.drop('education.num',axis=1)
df_test = df_test.drop('education.num',axis=1)

# dummy-encoded all the other categorical features
columns_to_dummy = ['marital.status', 'occupation',
           'relationship', 'race', 'sex', 'native.country']
df_train = pd.get_dummies(df_train, columns=columns_to_dummy, drop_first=True)
df_test = pd.get_dummies(df_test, columns=columns_to_dummy, drop_first=True)

# scale all the other numerical features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


std = StandardScaler()
mms = MinMaxScaler()
columns_to_scaler = ['capital.gain', 'capital.loss', 'hours.per.week']
df_train[columns_to_scaler] = std.fit_transform(df_train[columns_to_scaler])
df_test[columns_to_scaler] = std.fit_transform(df_test[columns_to_scaler])

# *********************** model starts here ***********************
# split
X_train = df_train.drop('income', axis=1)
X_test = df_test.drop('income', axis=1)
y_train = df_train['income']
y_test = df_test['income']


# model training and testing for gbc
gbc = GradientBoostingClassifier(n_estimators = 200, min_samples_split= 5, min_samples_leaf = 2, max_depth= 4, learning_rate = 0.1)
time_start = time.time()
gbc.fit(X_train, y_train)
time_end = time.time()
time_c = time_end - time_start
print('time cost for gbc', time_c, 's')
test_acc_3 = gbc.score(X_test, y_test)
print(f'GradientBoostingClassifier_Test accuracy: {test_acc_3:.3f}')
'''

# model training and testing for svm
svm = SVC()
time_start = time.time()
svm.fit(X_train, y_train)
time_end = time.time()
time_c = time_end - time_start
print('time cost for svm', time_c, 's')
test_acc_2 = svm.score(X_test, y_test)
print(f'SVM_Test accuracy: {test_acc_2:.3f}')


# model training and testing for lr
lr = LogisticRegression(max_iter=500)
time_start = time.time()
lr.fit(X_train, y_train)
time_end = time.time()
time_c = time_end - time_start
print('time cost for lr', time_c, 's')
test_acc_1 = lr.score(X_test, y_test)
print(f'LogisticRegression_Test accuracy: {test_acc_1:.3f}')
'''
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# model evaluating for gbc
y_pred = gbc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    ax.xaxis.set_ticks_position('bottom')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
'''
# for svm
y_pred_2 = svm.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_2)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    ax.xaxis.set_ticks_position('bottom')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# for lr
y_pred_3 = lr.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_1)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    ax.xaxis.set_ticks_position('bottom')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = y_test

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy_LR: {:.3f}".format(accuracy))

precision = precision_score(y_true, y_pred, pos_label='<=50K')
print("Precision_LR: {:.3f}".format(precision))

recall = recall_score(y_true, y_pred, pos_label='<=50K')
print("Recall_LR: {:.3f}".format(recall))

f1 = f1_score(y_true, y_pred, pos_label='<=50K')
print("F1-score_LR: {:.3f}".format(f1))

y_test_binary = (y_test == '<=50K').astype(int)
y_pred_1_binary = (y_pred == '<=50K').astype(int)

fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_1_binary)
auc_score = roc_auc_score(y_test_binary, y_pred_1_binary)

plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC Curve')
plt.legend()

plt.show()