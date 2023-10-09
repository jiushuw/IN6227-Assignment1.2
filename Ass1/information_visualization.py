import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
warnings.filterwarnings("ignore")

df = pd.read_csv("D:/AY23-24/IN6227DataMining/Census Income Data Set/adult_total.csv")

df.info()

# Info for numerical features
#df.describe()
# Info for categorical features
#df.describe(include=['O'])

duplicated_rows = df.duplicated()
any_duplicates = duplicated_rows.any()

print("Duplicated Rows:")
df[duplicated_rows]

df = df.drop_duplicates()

income_map = {'<=50K': 1, '>50K': 0}
df['income'] = df['income'].map(income_map)
df['income'] = df['income'].astype('int')

# DATA ANALYSIS AND FEATURE ENGINEERING
sns.histplot(df, x='age', hue='income', bins= 32)

def age_group(df):
    age_bins = [0, 20, 40, 60, float('inf')]
    age_labels = ['1', '2', '3', '4']
    df_age_range = df.copy()
    df_age_range['age'] = pd.cut(df_age_range['age'], bins=age_bins, labels=age_labels)
    return df_age_range

df = age_group(df).copy()

sns.histplot(df, x='age', hue='income', bins= 32)
age = pd.get_dummies(df['age'], drop_first=True)
df = df.drop('age',axis=1)
df = pd.concat([df,age],axis=1)
sns.countplot(data = df, x = 'workclass', hue = 'income')
df['workclass'] = df['workclass'].apply(lambda x: 1 if x == 'Private' else 0)
sns.countplot(data = df, x = 'workclass', hue = 'income')

df = df.drop('fnlwgt',axis=1)

sns.countplot(data = df, x = 'education', hue = 'income')
plt.tick_params(axis='x', rotation=90)

df = pd.get_dummies(df, columns=['education'], drop_first=True)
df = df.drop('education.num',axis=1)

# dummy-encoded all the other categorical features
columns_to_dummy = ['marital.status', 'occupation',
           'relationship', 'race', 'sex', 'native.country']
df = pd.get_dummies(df, columns=columns_to_dummy, drop_first=True)

# scale all the other numerical features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


std = StandardScaler()
mms = MinMaxScaler()
columns_to_scaler = ['capital.gain', 'capital.loss', 'hours.per.week']
df[columns_to_scaler] = std.fit_transform(df[columns_to_scaler])