import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor



train = pd.read_csv("train.csv")
y_train = train["Item_Outlet_Sales"]
X_train = train.drop("Item_Outlet_Sales", axis=1)
X_test = pd.read_csv("test.csv")

X_train_identifiers = X_train["Item_Identifier"].unique()
cat_attribs = X_train.select_dtypes(include=[object]).drop("Item_Identifier", axis=1).columns
num_attribs = X_train.select_dtypes(include=[np.number]).columns

for df in (X_train, X_test):
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({"low fat": "LF"})
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({"Low Fat": "LF"})
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({"Regular": "reg"})

for item in X_train_identifiers:
    sub_df = X_train.loc[X_train["Item_Identifier"] == item]
    for attrib in cat_attribs:
        if not(sub_df[attrib].isnull().values.all()):    
            X_train.loc[X_train["Item_Identifier"] == item, attrib] = X_train.loc[X_train["Item_Identifier"] == item, attrib].fillna(sub_df[attrib].mode()[0])
            X_test.loc[X_test["Item_Identifier"] == item, attrib] = X_test.loc[X_test["Item_Identifier"] == item, attrib].fillna(sub_df[attrib].mode()[0])
        else:
            X_train[attrib] = X_train[attrib].fillna(X_train[attrib].mode()[0])
            X_test[attrib] = X_test[attrib].fillna(X_train[attrib].mode()[0])
    for attrib in num_attribs:
        if not(sub_df[attrib].isnull().values.all()):
            X_train.loc[X_train["Item_Identifier"] == item, attrib] = X_train.loc[X_train["Item_Identifier"] == item, attrib].fillna(sub_df[attrib].median())
            X_test.loc[X_test["Item_Identifier"] == item, attrib] = X_test.loc[X_test["Item_Identifier"] == item, attrib].fillna(sub_df[attrib].median())
        else:
            X_train[attrib] = X_train[attrib].fillna(X_train[attrib].median())
            X_test[attrib] = X_test[attrib].fillna(X_train[attrib].median())

X_train.drop(["Item_Identifier"], axis=1, inplace=True) 
X_test.drop(["Item_Identifier"], axis=1, inplace=True) 

X_train["Item_Visibility"] = np.log1p(X_train["Item_Visibility"])
X_test["Item_Visibility"] = np.log1p(X_test["Item_Visibility"])
y_train = np.log1p(y_train)

cat_encoder = OneHotEncoder(handle_unknown="ignore")
X_train = cat_encoder.fit_transform(X_train)
X_test = cat_encoder.transform(X_test)

xscale = StandardScaler(with_mean=False)
yscale = StandardScaler(with_mean=False)
X_train = xscale.fit_transform(X_train)
y_train = yscale.fit_transform(np.array(y_train).reshape(-1,1))
X_test = xscale.transform(X_test)

print(X_train.shape)
print(y_train.shape)


kf = KFold(n_splits=10)

baseline_model_1 = CatBoostRegressor(verbose=0)
results1 = cross_val_score(baseline_model_1, X_train, y_train, scoring="neg_mean_squared_error", cv=kf)
print("\nCatBoostRegression:", results1)
# CatBoostRegression: [-0.49628265 -0.47859513 -0.49066155 -0.46758558 -0.45091981 -0.45435392
#  -0.48421676 -0.46355365 -0.44819811 -0.4749946 ]

# baseline_model_2 = LinearRegression()
# results2 = cross_val_score(baseline_model_2, X_train, y_train, scoring="neg_mean_squared_error", cv=kf)
# print("\nLinearRegression:", results2)
# # LinearRegression: [-0.66085688 -0.62942939 -0.65427829 -0.6104386  -0.58727243 -0.52266024
# #  -0.62110042 -0.63456533 -0.60034631 -0.63322657]

# baseline_model_3 = RandomForestRegressor(max_depth=5)
# results3 = cross_val_score(baseline_model_3, X_train, y_train.ravel(), scoring="neg_mean_squared_error", cv=kf)
# print("RandomForest:", results3)
# # RandomForest: [-0.57202255 -0.53917183 -0.55074252 -0.53686623 -0.50900986 -0.50679825
# #  -0.54943792 -0.52362875 -0.51040305 -0.54617588]

# def fit_poly(X, y, deg=1):
#     polynomial_features = PolynomialFeatures(degree=deg)
#     linear_regression = LinearRegression()
#     model = Pipeline([("poly_feat", polynomial_features), ("lin_reg", linear_regression)])
#     model.fit(X, y)
#     return model
# for d in [1,3,5,7,10,20]:
#     model = fit_poly(X_train, y_train, deg=d)
#     results = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
#     print("Polynomial Degree", d, results)

# for a in [10,100,1000,10000]:
#     for t in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]:
#         baseline_model = Ridge(alpha=a, tol=t)
#         results = cross_val_score(baseline_model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf)
#         print("a:", a, "t:", t, results)
