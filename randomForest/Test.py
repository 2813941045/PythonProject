from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# def datasets_demo():
from pandas import DataFrame    #加载pandas包
import pandas as pd
	#获取数据集
iris = load_iris()
x_data = iris.data       # .data返回iris数据集所有输入特征
y_data = iris.target     # .target返回iris数据集所有标签
                                         #用print函数打印出来看一下效果
print("x_data from datasets(未增加任何格式，直接显示数据): \n", x_data)
print("y_data from datasets(未增加任何格式，直接显示数据): \n", y_data)

x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
# 为增加可读性实用DataFrame()将数据转化成表格的形式，
# 每一列增加中文标签
# 为表格增加行索引（左侧）和列标签（上方）
pd.set_option('display.unicode.east_asian_width', True)
# 设置列对齐
print("x_data add index(每一列增加中文标签，设置列对齐): \n", x_data)

x_data['类别'] = y_data                            # 表格中新加一列，列标签为‘类别’，数据为y_data
print("x_data add a column(增加格式): \n", x_data)

#类型维度不确定时，建议用print函数打印出来确认效果
