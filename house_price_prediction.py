#Kütüphane İmportları

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
!pip install lightgbm
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
!pip install xgboost
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

#Set-Option Ayarları

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


#Veri Setlerinin Okutulması Ve Birleştirilmesi

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df = train.append(test, ignore_index=False).reset_index()
df.head()
df = df.drop("index", axis=1)


#Genel Resim İncelenmesi

def check_df(dataframe, head=5):
    print("################### SHAPE ###################")
    print(dataframe.shape)
    print("################### TYPES ###################")
    print(dataframe.dtypes)
    print("################### HEAD ###################")
    print(dataframe.head(head))
    print("################### TAIL ###################")
    print(dataframe.tail(head))
    print("################### NA ###################")
    print(dataframe.isnull().sum())
    print("################### QUANTİLES ###################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


#Nümerik ve Kategorik Değişkenlerin Yakalanması

def grab_col_names(dataframe, cat_th=10, car_th=20):
    #kategorik değişkenler
    cat_cols= [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                   and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                   and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [ col for col in cat_cols if col not in cat_but_car]

    #nümerik değişkenler
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations : {dataframe.shape[0]}")
    print(f"Variables : {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols, cat_but_car, num_cols

 cat_cols, cat_but_car, num_cols = grab_col_names(df)

df.columns
num_cols = [col for col in num_cols if col not in "Id"]


#Kategorik Değişken Analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : dataframe[col_name].value_counts() / len(dataframe) * 100}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
        plt.pause(5)

for col in cat_cols:
    cat_summary(df, col)


#Sayısal Değişken Analizi
def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        dataframe[col_name].hist(bins=50)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)
        plt.pause(5)

    print("###########################################")

for col in num_cols:
    num_summary(df, col)


#Hedef Değişken Analizi

def target_summary_w_cat(dataframe, target, cat_col):
    print(pd.DataFrame({"TARGET_MEAN" : dataframe.groupby(cat_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_w_cat(df, "SalePrice", col)

#bağımlı değişkenin incelenmesi
df["SalePrice"].hist(bins=100)
plt.show(block=True)
plt.pause(5)

#bağımlı değişkenin logaritmasının incelenmesi
np.log1p(df["SalePrice"]).hist(bins=50)
plt.show(block=True)
plt.pause(5)

#korelasyon analizi
corr = df[num_cols].corr()
corr
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)
plt.pause(5)


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block=True)
        plt.pause(5)
    return drop_list

high_correlated_cols(df, plot=False)
high_correlated_cols(df, plot=True)


#Feature Engineering

#aykırı değer analizi

def outlier_thresholds(dataframe, variable, q1 = 0.10, q3 = 0.90):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    interquartile = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * interquartile
    up_limit = quartile3 + 1.5 * interquartile
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

 for col in num_cols:
     if col != "SalePrice":
         print(col, check_outlier(df, col))


def replcae_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(df, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "SalePrice":
        replcae_with_thresholds(df, col)

for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df, col))


#eksik değerler

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / len(dataframe) * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n\n\n")
    if na_name:
        return na_columns


missing_values_table(df)

df["Alley"].value_counts()
df["BsmtQual"].value_counts()

no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

for col in no_cols:
    df[col].fillna("No", inplace=True)

missing_values_table(df)

def quick_missing_imp(dataframe, num_method="median", cat_length=20, target="SalePrice"):
    na_variables = [ col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    temp_target = dataframe[target]

    print("#### BEFORE ####")
    print(dataframe[na_variables].isnull().sum(), "\n\n")

    #Kategorik Değişkenler
    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    #Nümerik Değişkenler
    if num_method == "mean":
        dataframe = dataframe.apply(lambda x: x.fillna(x.mean()) if x.dtypes != "O" else x, axis=0)
    elif num_method == "median":
        dataframe = dataframe.apply(lambda x: x.fillna(x.median()) if x.dtypes != "O" else x, axis=0)

    dataframe[target] =temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(dataframe[na_variables].isnull().sum(), "\n\n")

    return dataframe

df = quick_missing_imp(df, num_method="median", cat_length=17)

# rare analizi ve rare encoder

def rare_anlyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":" ,len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT" : dataframe[col].value_counts(),
                            "RATIO" : dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN" : dataframe.groupby(col)[target].mean()}), end= "\n\n\n")

    rare_anlyser(df, "SalePrice", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])
        return temp_df

rare_encoder(df, 0.01)


#Yeni değişkenlerin oluşturulması

df["NEW_1st*GrLiv"] = df["1stFlrSF"] * df["GrLivArea"]

df["NEW_Garage*GrLiv"] = (df["GarageArea"] * df["GrLivArea"])

df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1)


# Total Floor
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]

# Total Finished Basement Area
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2

# Porch Area
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF

# Total House Area
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF

df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF


# Lot Ratio
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea

df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea #

df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea

# MasVnrArea
df["NEW_MasVnrRatio"] = df.MasVnrArea / df.NEW_TotalHouseArea

# Dif Area
df["NEW_DifArea"] = (df.LotArea - df["1stFlrSF"] - df.GarageArea - df.NEW_PorchArea - df.WoodDeckSF)


df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"]


df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt

df["NEW_HouseAge"] = df.YrSold - df.YearBuilt

df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd

df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt

df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd)

df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt



drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]

# drop_list'teki değişkenlerin düşürülmesi
df.drop(drop_list, axis=1, inplace=True)

df.columns
df.shape


#label encoding

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes =="O" and
              df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

#one-hot encoder

def one_hot_encoder(dataframe, cat_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.shape


#Modelleme

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = train_df['SalePrice'] # np.log1p(df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

reg_model.intercept_
reg_model.coef_

y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

R_square =  reg_model.score(X_test, y_test)
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))





























































































