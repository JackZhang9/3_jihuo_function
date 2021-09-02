#!/usr/bin/env python
# _*_coding: utf-8 _*_
# @Time : 2021/9/2 17:11
# @Author : CN-JackZhang
# @File: jihuo_function.py
import numpy as np
import matplotlib.pyplot as plt

#定义激活函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

x = np.arange(-10,10,0.1)

#sigmoid图像绘制
y_1 = sigmoid(x)
pic_sigmoid = plt.subplot(3,1,1)
pic_sigmoid.plot(x,y_1)
pic_sigmoid.set_title('sigmoid_function')
pic_sigmoid.axhline(y=0.5, ls='--',c='black')
pic_sigmoid.axvline(x=0, ls='--',c='black')

#relu图像绘制
y_2 = relu(x)
pic_relu = plt.subplot(3,1,2)
pic_relu.plot(x,y_2)
pic_relu.set_title('relu_function')
pic_relu.axvline(x=0,ls='--',c='black')

#tanh图像绘制
y_3 = tanh(x)
pic_tanh = plt.subplot(3,1,3)
pic_tanh.plot(x,y_3)
pic_tanh.set_title('tanh_function')
pic_tanh.axhline(y=0,ls='--',c='black')
pic_tanh.axvline(x=0,ls='--',c='black')


plt.show()
