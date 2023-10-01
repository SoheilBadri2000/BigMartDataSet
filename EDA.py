import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler


train = pd.read_csv("train.csv")


# MAKE SURE YOU ADJUST THE FAT CONTENT FEATURE VALUES !!!!!!!!!!!!!!!!!!
train["Item_Fat_Content"] = train["Item_Fat_Content"].replace({"low fat": "LF"})
train["Item_Fat_Content"] = train["Item_Fat_Content"].replace({"Low Fat": "LF"})
train["Item_Fat_Content"] = train["Item_Fat_Content"].replace({"Regular": "reg"})


# col = "Outlet_Type"
# print(train[col].value_counts())
# print(train[col].value_counts().shape)

train_identifiers = train["Item_Identifier"].unique()
cat_attribs = train.select_dtypes(include=[object]).drop("Item_Identifier", axis=1).columns
num_attribs = train.select_dtypes(include=[np.number]).columns

skew_df = pd.DataFrame(num_attribs, columns=["Feature"])
skew_df["Skew"] = skew_df["Feature"].apply(lambda feature: stats.skew(train[feature]))
skew_df["AbsSkew"] = skew_df["Skew"].apply(abs)
skew_df["Skewed"] = skew_df["AbsSkew"].apply(lambda x: True if x>= 0.5 else False)
print(skew_df)


for item in train_identifiers:
    sub_df = train.loc[train["Item_Identifier"] == item]
    for attrib in cat_attribs:
        if not(sub_df[attrib].isnull().values.all()):    
            train.loc[train["Item_Identifier"] == item, attrib] = train.loc[train["Item_Identifier"] == item, attrib].fillna(sub_df[attrib].mode()[0])
        else:
            train[attrib] = train[attrib].fillna(train[attrib].mode()[0])
    for attrib in num_attribs:
        if not(sub_df[attrib].isnull().values.all()):
            train.loc[train["Item_Identifier"] == item, attrib] = train.loc[train["Item_Identifier"] == item, attrib].fillna(sub_df[attrib].median())
        else:
            train[attrib] = train[attrib].fillna(train[attrib].median())



train.drop(["Item_Identifier"], axis=1, inplace=True)    

 
train = pd.get_dummies(train)

# train["Item_MRP"].hist(bins=50, figsize=(12,8))
train.plot.scatter(x="Item_Visibility", y="Item_Outlet_Sales")
plt.show()

print("\n\nOriginal")
skew_df = pd.DataFrame(num_attribs, columns=["Feature"])
skew_df["Skew"] = skew_df["Feature"].apply(lambda feature: stats.skew(train[feature]))
skew_df["AbsSkew"] = skew_df["Skew"].apply(abs)
skew_df["Skewed"] = skew_df["AbsSkew"].apply(lambda x: True if x>= 0.5 else False)
print(skew_df)

for column in skew_df.query("Skewed == True")["Feature"].values:
    train[column] = np.log1p(train[column])

print("\n\nAfter log")
skew_df = pd.DataFrame(num_attribs, columns=["Feature"])
skew_df["Skew"] = skew_df["Feature"].apply(lambda feature: stats.skew(train[feature]))
skew_df["AbsSkew"] = skew_df["Skew"].apply(abs)
skew_df["Skewed"] = skew_df["AbsSkew"].apply(lambda x: True if x>= 0.5 else False)
print(skew_df)

scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train), index=train.index, columns=train.columns)

print("\n\nAfter scaling")
skew_df = pd.DataFrame(num_attribs, columns=["Feature"])
skew_df["Skew"] = skew_df["Feature"].apply(lambda feature: stats.skew(train[feature]))
skew_df["AbsSkew"] = skew_df["Skew"].apply(abs)
skew_df["Skewed"] = skew_df["AbsSkew"].apply(lambda x: True if x>= 0.5 else False)
print(skew_df)

# train["Item_MRP"].hist(bins=50, figsize=(12,8))
train.plot.scatter(x="Item_Visibility", y="Item_Outlet_Sales")


# corr_matrix = train.corr(numeric_only=True)
# print(corr_matrix["Item_Outlet_Sales"].abs().sort_values(ascending=False))
# Item_Outlet_Sales                  1.000000
# Item_MRP                           0.567574
# Outlet_Type_Grocery Store          0.411727
# Outlet_Type_Supermarket Type3      0.311192
# Outlet_Identifier_OUT027           0.311192
# Outlet_Identifier_OUT010           0.284883
# Outlet_Identifier_OUT019           0.277250
# Item_Visibility                    0.128625
# Outlet_Location_Type_Tier 1        0.111287
# Outlet_Type_Supermarket Type1      0.108765
# Outlet_Size_Small                  0.098403
# Outlet_Size_Medium                 0.075154
# Outlet_Location_Type_Tier 2        0.058261
# Outlet_Identifier_OUT035           0.052823
# Outlet_Establishment_Year          0.049135
# Outlet_Location_Type_Tier 3        0.046376
# Item_Type_Baking Goods             0.038381
# Outlet_Type_Supermarket Type2      0.038059
# Outlet_Identifier_OUT018           0.038059
# Outlet_Identifier_OUT049           0.034264
# Outlet_Identifier_OUT017           0.032610
# Item_Type_Fruits and Vegetables    0.025950
# Item_Type_Health and Hygiene       0.025587
# Outlet_Size_High                   0.024170
# Outlet_Identifier_OUT013           0.024170
# Item_Type_Soft Drinks              0.024040
# Item_Type_Snack Foods              0.022782
# Item_Type_Others                   0.021267
# Outlet_Identifier_OUT046           0.019803
# Item_Fat_Content_LF                0.018719
# Item_Fat_Content_reg               0.018719
# Item_Type_Household                0.015701
# Item_Type_Starchy Foods            0.015039
# Item_Weight                        0.009693
# Item_Type_Frozen Foods             0.009482
# Item_Type_Dairy                    0.008858
# Item_Type_Canned                   0.007387
# Item_Type_Seafood                  0.007380
# Item_Type_Breakfast                0.004656
# Item_Type_Hard Drinks              0.003956
# Item_Type_Meat                     0.002995
# Item_Type_Breads                   0.002332
# Outlet_Identifier_OUT045           0.0022744

plt.show()