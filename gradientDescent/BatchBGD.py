#代码规范化，用函数包装成代码块，不要散着写
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score

#1.首先读取数据，写一个读取数据的函数
def getData():
    data=np.loadtxt("C:/Users/小风风/Desktop/boston_housing_data.csv",delimiter=',',skiprows=1,dtype=np.float32)
    X=data[:,1:]  #二维矩阵
    Y=data[:,0]  #打印结果为一行
    Y=Y.reshape(-1,1) #将y的一行转为一列二维矩阵，行数由列数自动计算
    X=dataNumalize(X) #归一化之后的X
    return X,Y

#2.将特征集数据进行归一化处理，处理后的数据也是要进行输出，所以此函数要在读取数据的函数中进行调用
#归一化是对一个矩阵进行归一化，所以此函数有一个参数
#归一化使用（某个值-均值）/标准差
def dataNumalize(X):
    mu=X.mean(0)  #对X特征集的第一列的均值 X(n)
    std=np.std(X)  #S(n)
    X=(X-mu)/std   #将归一化处理结果赋值X
    index=np.ones((len(X),1)) #最后一列1
    X=np.hstack((X,index)) #进行合并 ，此合并函数只含一个参数，所以将X，index带个括号看成一个参数
    return X

#3.下面计算损失函数，在损失函数中有3个参数：theta,X,Y
def lossFuncation(X,Y,theta):
    m=X.shape[0]  #m是X特征集的样本数，即行数
    loss=sum((np.dot(X,theta)-Y)**2)/(2*m)
    return loss

#4.下面实现批量下降梯度算法
#此算法的theta更新公式中有四个参数，X,Y,theta,alpha
#另外我们需要定义一个迭代次数num_iters，就是更新多少次（alpha走多少步）才能下降到最低点，每迭代一次，theta就更新一次
def BGD(X,Y,theta,alpha,num_iters):
    m=X.shape[0]
    loss_all=[]  #定义loss值列表，把每次theta更新时跟着变动的loss函数值存入展示
    for i in range(num_iters):
        theta=theta-alpha*np.dot(X.T,np.dot(X,theta)-Y)/m
        loss=lossFuncation(X,Y,theta)
        loss_all.append(loss)  #吧loss值添加到loss_all列表中去
        print("第{}次的loss值为{}".format((i+1),loss))
    return theta,loss_all

#5.主函数进行测试
#此测试没有区分text和predict，所有的数据用来做测试集
if __name__=='__main__':
    X,Y=getData()
    theta=np.ones((X.shape[1],1))  #对theta起点值进行初始化
    num_iters=1000  #初始化迭代次数为500
    alpha=0.01  #初始化α（下降步长）为0.01
    theta,loss_all=BGD(X,Y,theta,alpha,num_iters) #调用BGD函数求theta更新参数，和每次更新后的loss函数值
    print(theta)
    #做模型预测
    y_predict=np.dot(X,theta)



#最后进行一个模型的评价----根据模型评价指标与所的结果进行判定,结果与迭代次数有关
print("平均绝对误差：",mean_absolute_error(Y,y_predict))
print("均方误差：",mean_squared_error(Y,y_predict))
print("中之绝对误差：",median_absolute_error(Y,y_predict))
print('r2',r2_score(Y,y_predict))

#模型测试完毕，模型预测结果进行数据可视化
plt.scatter(np.arange(100),Y[:100],c='red')  #真实label
plt.scatter(np.arange(100),y_predict[:100],c='green') #预测的label
plt.show()
#绘制损失函数随着迭代次数改变所进行变化曲线
plt.plot(np.arange(num_iters),loss_all,c='black')
#plt.show()
