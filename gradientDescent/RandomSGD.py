#随机算法代码SGD有两种写法，
#1.步骤和批量梯度一样，只需要把数据集改为某一个数据就可以
#2.用sklearn库实现随机梯度下降算法，此代码我们使用sklearn实现
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #sklearn库中用来归一化的
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error

#1.也是要一开始读入数据
data=np.loadtxt("C:/Users/小风风/Desktop/boston_housing_data.csv",delimiter=',',skiprows=1,dtype=np.float32)
#分片，切割数据
X=data[:,1:]
Y=data[:,0]
#数据分为测试集和训练集
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.8,random_state=42)

#2.数据预处理，把数据归一化
scaler=StandardScaler()
scaler.fit(X_train) #将训练集归一化，计算均值和方差
X=scaler.transform(X_train) #将归一化结果赋值给X

#3.计算损失函数，sklearn库会自动计算该函数
#4.训练模型：使用sklearn中的SGD随机算法更新参数，训练模型
sgd=SGDRegressor()
sgd.fit(X_train,Y_train)  #使用训练集进行训练
y_predict=sgd.predict(X_test) #使用训练好的模型进行预测
print(y_predict)

#5.模型可视化
plt.scatter(np.arange(100),Y[:100],c='red')
plt.scatter(np.arange(100),y_predict[:100],c='black')
plt.show()

#6.模型的评价，评价方法解析解和sklearn都是一样的
print("平均绝对误差：",mean_absolute_error(Y_test,y_predict))
print("均方误差：",mean_squared_error(Y_test,y_predict))
print("中之绝对误差：",median_absolute_error(Y_test,y_predict))

#SGD算法速度快但准确性低，
