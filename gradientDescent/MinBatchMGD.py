import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score

#读数据
def getData():
    data=np.loadtxt("C:/Users/小风风/Desktop/boston_housing_data.csv",delimiter=',',skiprows=1,dtype=np.float32)
    X=data[:,1:]
    Y=data[:,0]
    Y=Y.reshape(-1,1)
    X=dataNumalize(X)
    return X,Y
#数据归一化处理
def dataNumalize(X):
    mu=X.mean(0)
    std=np.std(X)
    X=(X-mu)/std
    index=np.ones((X.shape[0],1))
    X=np.hstack((X,index))
    return X
#计算损失函数---3个参数
def lossFunction(X,Y,theta):
    m=X.shape[0]
    loss=sum((np.dot(X,theta)-Y)**2)/(2*m)
    return loss
#实现MGD更新theta--五个参数--要进行测试的X，Y，theta,alpha,还有迭代次数
#值得注意的是，这里我们用的测试数据是从X中随机抽取的500个小批量样本
def MGD(X,Y,theta,alpha,num_iters):
    m=X.shape[0]
    loss_all=[]
    for i in range(num_iters):
        index = np.random.choice(a=np.arange(0,m), size=1000, replace=False)
        x_new = X[index]
        y_new = Y[index]
        theta = theta - alpha * np.dot(x_new.T, (np.dot(x_new, theta) - y_new)) / m
        #theta=theta-alpha*np.dot(x_new.T,(np.dot(x_new,theta)-y_new))/m
        loss=lossFunction(x_new,y_new,theta)
        loss_all.append(loss)
        print("第{}次的loss值为{}".format(i+1,loss))
    return theta,loss_all

#主函数测试
if __name__=='__main__':
    X,Y=getData()
    theta=np.ones((X.shape[1],1)) #???
    num_iters=1000
    alpha=0.01
    theta,loss_all=MGD(X,Y,theta,alpha,num_iters)
    print(theta)
    #模型进行数据预测
    y_predict=np.dot(X,theta)

#模型的评价
print("均值误差：",mean_absolute_error(Y,y_predict))
print("mse", mean_squared_error(Y, y_predict))
print("median-ae", median_absolute_error(Y, y_predict))
print("r2", r2_score(Y, y_predict))

#数据结果可视化
plt.scatter(np.arange(100),Y[:100],c='red')
plt.scatter(np.arange(100),y_predict[:100],c='black')
#plt.plot(np.arange(num_iters),loss_all,c='black')
plt.show()

