import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


import warnings
warnings.simplefilter(action="ignore")



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("datasets/Telco-Customer-Churn.csv")


# Adım 1: Genel resmi inceleyiniz.
def check_df(dataframe, head=5):
    print("####### Shape ########")
    print(dataframe.shape)
    print("####### Types ########")
    print(dataframe.dtypes)
    print("####### Head #########")
    print(dataframe.head(head))
    print("######## Tail ###########")
    print(dataframe.tail(head))
    print("####### NA #########")
    print(dataframe.isnull().sum())
    print("###### Quantiles ####")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)
df["TotalCharges"].dtypes
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f"cat_cols : {dataframe[cat_cols].columns.values}")
    print(f"num_cols : {dataframe[num_cols].columns.values}")
    print(f"num_but_cat : {dataframe[num_but_cat].columns.values}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)


for col in num_cols:
    num_summary(df, col)



# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

def target_sum_with_num(df, target, numerical_col):
    print(df.groupby(target).agg({numerical_col: "mean"}))

for col in num_cols:
    target_sum_with_num(df, "Churn", col)


def target_sum_with_cat(df, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": df.groupby(categorical_col)[target].mean(),
                        "Count": df[categorical_col].value_counts(),
                        "Ratio": 100 * df[categorical_col].value_counts() / len(df)}), end="\n\n\n")

for col in cat_cols:
    target_sum_with_cat(df, "Churn", col)

# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



# Adım 6: Eksik gözlem analizi yapınız.
df.isnull().sum()
#eksik değer sadece "TotalCharges" değişkeninde olduğu için ve adeti 11 olduğu için drop edilir.
#

df.isnull().sum()
df.shape

# x = [col for col in df.columns if df["TotalCharges"].isnull() == True]

# list= []
# a= np.array(list)
# for value in df["TotalCharges"]:
#     if df["TotalCharges"].isnull() != True :
#         continue
#     else:
#         a= np.append(a,value)
#         print(a)

# Adım 7: Korelasyon analizi yapınız.

df.corr()

# Görev 2 : Feature Engineering
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

df = df[~df["TotalCharges"].isnull()]    ##varolan 11 değer drop edilir.


for col in num_cols:
    print(col, check_outlier(df, col)) ###aykırı değerler yoktur.
#
# for col in df.columns:
#     print(col, check_outlier(df, col))
#     if check_outlier(df,col):
#         replace_with_thresholds(df,col)





# Adım 2: Yeni değişkenler oluşturunuz.

# Adım 3: Encoding işlemlerini gerçekleştiriniz.

binary_cols = [col for col in cat_cols if df[col].dtypes == "O" and df[col].nunique()==2]


## 1-0 olan kategorik değişkenler için label encoding
def laberencoder(df, cols):
    laberencoding = LabelEncoder()
    df[cols] = laberencoding.fit_transform(df[cols])
    return df

for col in binary_cols:
    df = laberencoder(df,col)

df.head()
#2 den fazla sınıf içeren kategorik değişkenlerin vektör dizeleri halinde temsili için için one-hot encoding
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn"]]


def one_hot_encoding(df,cols, drop_first = False):
    df = pd.get_dummies(df,columns = cols,drop_first = drop_first)
    return df


df =one_hot_encoding(df, cat_cols, drop_first=True)

df


###one hot encoding ordinal olmayan kategorik değişkenler için, integer encoding ise, ordinal olan 2 den fazla sınıfı olan kategor,ik değişkenler için


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


df.head()
# Adım 5: Model oluşturunuz.


y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

df.head()