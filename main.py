import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Get the data
FilePath =r'D:\OneDrive - University Of Jordan\Desktop\newProject\diamonds.csv'
diamond = pd.read_csv(FilePath)

#look at the big picture
print(diamond.head())
print(diamond.info())

# Discover and visualize the data
diamond.drop(["Unnamed: 0"], axis=1,inplace=True)
print(diamond.isnull().sum())
print(diamond.describe())
# 20 elements deleted
print(diamond.shape)

print(diamond['x'].min())
print(diamond['y'].min())
print(diamond['z'].min())
print(len(diamond[(diamond['x']==0) | (diamond['y']==0) | (diamond['z']==0)]))

plt.bar(diamond['cut'].unique(),diamond['cut'].value_counts())
plt.show()
plt.bar(diamond['clarity'].unique(),diamond['clarity'].value_counts())
plt.show()

plt.bar(diamond['color'].unique(),diamond['color'].value_counts())
plt.show()

plot = sns.pairplot(diamond,hue='cut',palette='bright')
plt.show()

sns.set_theme(color_codes=True)
plot=sns.regplot(x='price',y='x',data=diamond,fit_reg=True,color="blue" ,line_kws={'color':'k'})
plt.show()
plot=sns.regplot(x='price',y='y',data=diamond,fit_reg=True,line_kws={'color':'k'})
plt.show()
plot=sns.regplot(x='price',y='z',data=diamond,fit_reg=True,color='orange',line_kws={'color':'k'})
plt.show()
plot=sns.regplot(x='price',y='depth',data=diamond,fit_reg=True,color='green',line_kws={'color':'k'})
plt.show()
plot=sns.regplot(x='price',y='table',data=diamond,fit_reg=True,color='red',line_kws={'color':'k'})
plt.show()


diamond = diamond[(diamond["depth"]<75)&(diamond["depth"]>45)]
diamond = diamond[(diamond["table"]<80)&(diamond["table"]>40)]
diamond = diamond[(diamond["x"]<30)]
diamond = diamond[(diamond["y"]<30)]
diamond = diamond[(diamond["z"]<30)&(diamond["z"]>2)]
diamond.shape # 13 element deleted

# For categorical data
plot=sns.boxplot(x='price',y='cut',palette='bright',data=diamond)
plt.show()
plot=sns.boxplot(x='price',y='color',palette='bright',data=diamond)
plt.show()
plot=sns.boxplot(x='price',y='clarity',palette='bright',data=diamond)
plt.show()

# Prepare the data for machine learning algorithms
diamond["newDistance"] = diamond['x']*diamond['y']*diamond["z"]

diamond["volume"] = (diamond["carat"]*0.2)/ 3.51
diamond["lenPerCarat"] = diamond["x"]/diamond["carat"]

color_cat = diamond[["color"]]
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
color_cat_encoded = ordinal_encoder.fit_transform(color_cat)
np.unique(color_cat_encoded)
diamond["newcolor"] = color_cat_encoded

diamond_cat = diamond[["cut","clarity"]]
ordinal_encoder = OrdinalEncoder(categories = [['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                                              ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']])
diamond_cat_encoded = ordinal_encoder.fit_transform(diamond_cat)
clarity_cat_encoder = diamond_cat_encoded[0:,1]
cut_cat_encoded = diamond_cat_encoded[0:,0]

diamond["newcut"] = cut_cat_encoded
diamond["newclarity"] = clarity_cat_encoder

corr_dia = diamond.corr()
print(corr_dia["price"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attribute = ["carat",	"cut",	"color",	"clarity",	"depth",	"table",	"price","volume",	"x",	"y",	"z"]
scatter_matrix(diamond[attribute],figsize=(12,8))
plt.show()

diamond.plot(kind="scatter",x="carat",y="price",alpha=0.1)
plt.show()
# the correlation between price and carat is very strong ,
# the point is not too dispersed and some how there is no outlires

diamond.plot(kind="scatter",x="carat",y="x",alpha=0.1)
plt.show()
diamond.plot(kind="scatter",x="lenPerCarat",y="price",alpha=0.1)
plt.show()

print(diamond.describe())
print(diamond['x'].min())
print(diamond['y'].min())
print(diamond['z'].min())
# there is no missing vlaue ,So the data set is clean

diamond.drop(["cut","color","clarity"],axis=1,inplace=True)
plt.show()
print(diamond.head())


attribute=["newcolor","newcut","newclarity"]
diamondNum=diamond.drop(attribute,axis=1)
scaler = StandardScaler()
scalerDiamond=scaler.fit_transform(diamondNum)
print(scalerDiamond[:10])

# select and train model
x=diamond.drop(['price'],axis=1)
y=diamond['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
print(lin_reg.fit(x_train, y_train))

from sklearn.metrics import mean_squared_error
diamond_predictions = lin_reg.predict(x_train)
lin_mse = mean_squared_error(y_train, diamond_predictions)
lin_rmse = np.sqrt(lin_mse)

print(lin_rmse)
print(lin_reg.score(x_train,y_train))

from sklearn import linear_model
reg = linear_model.BayesianRidge()
print(reg.fit(x_train,y_train))

print(reg.score(x_train,y_train))

from sklearn.metrics import mean_squared_error
diamond_predictions = reg.predict(x_train)
reg_mse = mean_squared_error(y_train, diamond_predictions)
reg_rmse = np.sqrt(reg_mse)
print(reg_rmse)

from sklearn.linear_model import Lasso
lasso_reg = Lasso()
lasso_reg.fit(x_train, y_train)
diamondLasso_predictions = lasso_reg.predict(x_train)
lasso_mse = mean_squared_error(y_train, diamondLasso_predictions)
lasso_rmse = np.sqrt(lasso_mse)
print(lasso_rmse)

print(lasso_reg.score(x_test,y_test))

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
diamondTree_predictions = tree_reg.predict(x_train)
tree_mse = mean_squared_error(y_train, diamondTree_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

print(tree_reg.score(x_train,y_train))

print(tree_reg.predict(x_train[:5]))
print(np.array(y_train[:5]))

print(tree_reg.predict(x_test[:5]))
print(np.array(y_test[:5]))

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(x_train, y_train)
diamondForest_predictions = forest_reg.predict(x_train)
forest_mse = mean_squared_error(y_train, diamondForest_predictions)
forest_rmse = np.sqrt(forest_mse)

print(forest_rmse)
print(forest_reg.score(x_train,y_train))

print(forest_reg.predict(x_train[:5]))
print(np.array(y_train[:5]))

# fine-tune the model

from sklearn.model_selection import cross_val_score

scores = cross_val_score(forest_reg, x_train, y_train,scoring="neg_mean_squared_error", cv=10)
forestReg_rmse_scores = np.sqrt(-scores)

print(forestReg_rmse_scores)

def display_scores(scores):
  print("Scores:", scores)
  print("Mean:", scores.mean())
  print("Standard deviation:", scores.std())

print(display_scores(forestReg_rmse_scores))
