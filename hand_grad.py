# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#############set params############
lr=1e-4
epoches=200
####################################
#################training#########
def get_grad(w,x,y):
    result=[]
    for i in range(w.size):
        result.append(0)
    wx=w.dot(x.T)
    if wx<0:
        return np.array(result)
    for i in range(w.size):
        result[i]=2*(wx-y)*x[i]#梯度计算公式
    return np.array(result)

df=pd.read_csv("D:\\MyDownload\\train.csv")
df=df.loc[1:,['open','high','low','wave','close']]
############归一化#######
df = (df - df.mean()) / df.std()
########################
train_y=df['close']
train_x=df[['open','high','low','wave']]
w=np.array([0,0,0,0])
train_x=train_x.values
train_y=train_y.values
while epoches:
    epoches-=1
    for i in range(len(train_x)):
        x=train_x[i]
        grad_w=get_grad(w,x,train_y[i])#计算梯度
        w=w-lr*grad_w#反向传播给w
print('最终的权重矩阵为',w)
###############################################3
###################testing#################
df=pd.read_csv("D:\\MyDownload\\test.csv")
df=df.loc[:,['open','high','low','wave','close']]
result=df['close'].values
test_x=df[['open','high','low','wave']]
test_x=test_x.values
test_y=[]
for i in range(len(test_x)):
    x=test_x[i]
    test_y.append(w.dot(x.T))
test_y=np.array(test_y)
print('预测结果为',test_y)
print('误差',abs(result-test_y))
plt.plot(result,'r')
plt.plot(test_y,'b' )
plt.show()

