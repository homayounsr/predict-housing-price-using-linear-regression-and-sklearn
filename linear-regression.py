import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('datasets/house.csv')


#[x]spreate data to x(feature) and y(target) array
x=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y=df['Price']
#[x]split train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=101)
#[x]import LR
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
#[x]train the model on training data
lm.fit(x_train,y_train)
#[x]## Model Evaluation | Let's evaluate the model by checking out it's coefficients and how we can interpret them.
print(lm.intercept_)
print(lm.coef_)
#[x]Display Dataframe based on coeffcient
df_coeff=pd.DataFrame(lm.coef_,x.columns,columns=['coeff'])
print(df_coeff)
#[x]grabbing prediction from our test set
predictions = lm.predict(x_test)
#[x]shoing prediction on scatter plot
plt.scatter(y_test,predictions)
plt.show()
#[x]create a hist of the dist of our residual
sns.distplot((y_test-predictions))
plt.show()
print(x_test.columns)
#[x]regression evaluation matrics
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
