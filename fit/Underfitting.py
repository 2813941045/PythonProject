from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures  # 用于解决欠拟合的问题的多项式回归

# 自己定义训练数据
x_train = [[6], [8], [10], [14], [18]]  # 大小
y_train = [[7], [9], [13], [17.5], [18]]  # 价格

# 进行阶数为一阶的线性回归和预测
linear = LinearRegression()
linear.fit(x_train, y_train)

# 绘制基于出原始样本数据得出来的拟合曲线 + 散点图
xx = np.linspace(0, 25, 100)
xx = xx.reshape(-1, 1)
yy = linear.predict(xx)
plt.scatter(x_train, y_train)
plt.plot(xx, yy)
plt.show()


"""多项式回归解决出现的欠拟合问题"""

# 建立二次多项式线性回归模型进行预测
poly2 = PolynomialFeatures(degree=2)  # 2次多项式特征生成器
x_train_poly2 = poly2.fit_transform(x_train)
# 建立模型预测
linear2_poly2 = LinearRegression()
linear2_poly2.fit(x_train_poly2, y_train)

# 绘制基于多项式回归后得出来的拟合曲线 + 散点图
xx_poly2 = poly2.transform(xx)
yy_poly2 = linear2_poly2.predict(xx_poly2)
plt.scatter(x_train, y_train)
plt.plot(xx, yy, label="Degree = 1")
plt.plot(xx, yy_poly2, label="Degree = 2")
plt.legend()
plt.show()

"""使用模型对未知价格的蛋糕进行价格的预测"""
x_test1 = np.array([15]).reshape(1, -1)
x_test1_poly2 = poly2.fit_transform(x_test1.reshape(1, -1))
y_test1 = linear2_poly2.predict(x_test1_poly2)
print(y_test1)
