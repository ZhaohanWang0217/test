#iweek4_1:搭建卷积神经网络模型以及应用
'''
        共两个部分：
        1.神经网络的底层搭建
        2.神经网络的应用
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader,TensorDataset
from torch import nn   #nn是专门为神经网络设计的模块化接口，可以用来定义和运行神经网络。
import numpy as np
from matplotlib import pyplot as plt
from cnn_utils import load_dataset

#%matplotlib inline  #功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。
plt.rcParams['figure.figsize'] = (5.0,4.0)
plt.rcParams['image.interpolation'] = 'nearest'#设置插值风格，最近邻差值，像素是正方形
plt.rcParams['image.cmap'] = 'gray'#设置颜色，使用灰度输出而不是彩色输出

#%load_ext autoreload

#%autoreload 2

np.random.seed(1)   #指定随机种子


#神经网络的底层搭建
print("==========Step1:神经网络的底层搭建===========")
'''
    我们要实现一个拥有卷积层（conv）和池化层（pool）的网络，包含了前向和反向传播。
    1.卷积模块，包含了以下函数：
        （1）.使用0扩充边界
        （2）.卷积窗口
        （3）.前向卷积
        （4）.反向卷积
    2.池化模块，包含了以下函数
        （1）.前向池化
        （2）.创建掩码
        （3）.值分配
        （4）.反向池化
'''


#扩充边界
print("**********1.1:扩充边界************")
#constant连续一样的值填充，有constant_values=（x，y）时前面用x填充，后面用y填充。缺省参数是为constant_values=(0,0)

#a = np.pad(a,((0,0),(1,1),(0,0),(3,3),(0,0)),'constant',constant_values = (...,...))

#比如：
arr3D = np.array([[[1,1,2,2,3,4],
                   [1,1,2,2,3,4],
                   [1,1,2,2,3,4]],
                 [[0,1,2,3,4,5],
                  [0,1,2,3,4,5],
                  [0,1,2,3,4,5]],
                 [[1,1,2,2,3,4],
                  [1,1,2,2,3,4],
                  [1,1,2,2,3,4]]])
#测试以下
#print('constant: \n' + str(np.pad(arr3D,((0,0),(1,1),(2,2)),'constant')))

def zero_pad(X,pad):
    '''
        把数据集X的图像边界全部使用0来扩充pad个宽度和高度
    :param X: 图像数据集，维度为（样本数，图像高度，图像宽度，图像通道数）
    :param pad: 整数，每个图像在垂直和水平维度上的填充量
    :return:
            X_paded:扩充后的图像数据集，维度为（样本数，图像高度+2*pad,图像宽度+2*pad,图像通道数）
    '''

    X_paded = np.pad(X,(
        (0,0),   #样本数，不填充
        (pad,pad), #图像高度，你可以视为上面填充x个，下面填充y个，，（x,y）
        (pad,pad),#图像宽度，你可以视为左边填充x个，右边填充y个，，（x,y）
        (0,0)     #通道数，不填充
    ),
                     'constant',constant_values=0)#连续一样的值填充，填充数字为0

    return X_paded

#测试一下
X = np.random.randn(4,3,3,2)
X_paded = zero_pad(X,2)
print("X.shape = ",X.shape)
print("X_paded.shape = ",X_paded.shape)
print('X[1,1] = ',X[1,1])
print("X_paded[1,1] = ",X_paded[1,1])

#绘制图
fig,axarr = plt.subplots(1,2) #一行两列,,fig画窗的意思，ax是画窗中创建的笛卡尔坐标区
axarr[0].set_title('X')
axarr[0].imshow(X[0,:,:,0])
axarr[1].set_title('X_paded')
axarr[1].imshow(X_paded[0,:,:,0])
plt.show()


#单步卷积
print("**********1.2:单步卷积************")

def conv_single_step(a_slice_prev,W,b):
    '''
            在前一层的激活输出的一个片段上应用一个由参数W定义的过滤器。
        这里切片大小和过滤器大小相同。
    :param a_slice_prev:输入数据的一个片段，维度为（过滤器大小，过滤器大小，上一通道数）
    :param W:权重参数，包含在了一个矩阵中，维度为（过滤器大小，过滤器大小，上一通道数）
    :param b:偏置参数，包含在了一个矩阵中，维度为（1，1，1）
    :return:
            Z：在输入数据的片X上卷积滑动窗口（W，b）的结果。
    '''

    s = np.multiply(a_slice_prev,W) + b#对应元素相乘

    Z = np.sum(s)

    return Z

#测试一下
np.random.seed(1)
a_slice_prev = np.random.randn(4,4,3)
W = np.random.randn(4,4,3)
b = np.random.randn(1,1,1)

Z = conv_single_step(a_slice_prev,W,b)
print("Z = " + str(Z))

#卷积神经网络-前向传播
print("**********1.3:前向传播************")
def conv_forward(A_prev,W,b,hparameters):
    '''
    实现卷积函数的前向传播
    :param A_prev: 上一层的激活输出矩阵，维度为（m,n_H_prev,n_W_prev,n_C_prev）
    :param W: 权重矩阵，维度为（f,f,n_C_prev,n_C）
    :param b: 偏置矩阵，维度为（1，1，1，n_C）,(1,1,1，这一层的过滤器数量)
    :param hparameters: 包含了”stride“与’pad‘的超参数字典。
    :return:
            Z：卷积输出，维度为（m,n_H,n_W,n_C）
            cache:缓存了一些反向传播函数conv_backward()需要的一些数据
    '''

    #获取来自上一层数据的基本信息
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    #获取权重矩阵的基本信息
    (f,f,n_C_prev,n_C) = W.shape

    #获取超参数hparameters的值
    stride = hparameters["stride"]
    pad = hparameters['pad']

    #计算卷积后的图像的宽度高度
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    #使用0来初始化卷积输出z
    Z = np.zeros((m,n_H,n_W,n_C))

    #通过A_prev创建填充过了的A_prev_pad
    A_prev_pad = zero_pad(A_prev,pad)

    for i in range(m):                           #遍历样本
        a_prev_pad = A_prev_pad[i]               #选择第i个样本的扩充后的激活矩阵
        for h in range(n_H):                     #在输出的垂直轴上循环
            for w in range(n_W):                 #在输出的水平轴上循环
                for c in range(n_C):             #循环遍历输出的通道
                    #定位当前的切片位置
                    vert_start = h * stride      #竖向，开始的位置
                    vert_end = vert_start + f    #竖向，结束的位置
                    horiz_start = w * stride     #横向，开始的位置
                    horiz_end = horiz_start + f  #横向，结束的位置

                    #切片位置定位好了我们就把它取出来，需要注意的是我们是”穿透“取出来的。
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    #执行单步卷积
                    Z[i,h,w,c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[0,0,0,c])#未加激活函数

    #数据处理完毕，验证数据格式是否正确
    assert (Z.shape == (m,n_H,n_W,n_C))

    #存储一些缓存值，以便于反向传播使用
    cache = (A_prev,W,b,hparameters)

    return (Z,cache)

#测试一下
np.random.seed(1)

A_prev = np.random.randn(10,4,4,3)
#print(A_prev)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)

hparameters = {"pad":2,"stride":1}

Z,cache_conv = conv_forward(A_prev,W,b,hparameters)

print("np.mean(Z) = ",np.mean(Z))
print("cache_conv[0][1][2][3] = ",cache_conv[0][1][2][3])


#池化层-前向传播
print("**********1.4:池化层************")

def pool_forward(A_prev,hparameters,mode='max'):
    '''
       实现池化层的前向传播
    :param A_prev: 输入数据，维度为(m,n_H_prev,n_W_prev,n_C_prev)
    :param hparameters:包含了”f“和”stride“的超参数字典
    :param mode: 模式选择”max“和”average“
    :return:
            A : 池化层的输出，维度为（m,n_H,n_W,n_C）
            cache:存储了一些反向传播需要用到的值，包含了输入和超参数的字典
    '''

    #获取输入数据的基本信息
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    #获取超参数的信息
    f = hparameters["f"]
    stride = hparameters['stride']

    #计算输出维度
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    #初试化输出矩阵
    A = np.zeros((m,n_H,n_W,n_C))

    for i in range(m):                 #遍历样本
        for h in range(n_H):           #在输出的垂直轴上循环
            for w in range(n_W):       #在输出的水平轴上循环
                for c in range(n_C):   #循环遍历输出的通道
                    #定位当前的切片位置
                    vert_start = h * stride            #竖向，开始的位置
                    vert_end = vert_start + f          #竖向，结束的位置
                    horiz_start = w *stride            #横向，开始的位置
                    horiz_end = horiz_start + f        #横向，结束的位置

                    #定位完毕，开始切割
                    a_slice_prev = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]

                    #对切片进行池化操作
                    if mode == "max":
                        A[i,h,w,c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i,h,w,c] = np.mean(a_slice_prev)

    #池化完毕，校验数据格式
    assert (A.shape == (m,n_H,n_W,n_C))

    #校验完毕，开始存储用于反向传播的值
    cache = (A_prev,hparameters)

    return A,cache

#测试一下
np.random.seed(1)
A_prev = np.random.randn(2,4,4,3)
hparameters = {"f":4,"stride":1}

A,cache = pool_forward(A_prev,hparameters,mode="max")
#A,cache = pool_forward(A_prev,hparameters)
print("A = ",A)
print(A.shape)
print('------------------')
A,cache = pool_forward(A_prev,hparameters,mode="average")
print("A = ",A)

#卷积神经网络中的反向传播
print("**********1.5:卷积神经网络中的反向传播************")
'''
    在现在的深读学习框架中，你只需要实现前向传播，框架负责向后传播，卷积网络的反向传播有点复杂。
'''
print("----------1.5.1:卷积层反向传播------------")

def conv_backward(dZ,cache):
    '''
    实现卷积层的反向传播
    :param dZ: 卷积层的输出Z的梯度，维度为（m,n_H,n_W,n_C）
    :param cache: 反向传播所需要的参数，conv_forward()的输出之一
    :return:
            dA_prev:卷积层的输入（A_prev）的梯度值，维度为（m,n_H_prev,n_W_prev,n_C_prev）
            dW : 卷积层的权值的梯度，维度为（f,f,n_C_prev,n_C）
            db : 卷积层的偏置的梯度，维度为（1，1，1，n_C）
    '''

    #获取cache的值
    (A_prev,W,b,hparameters) = cache

    #获取A_prev的基本信息
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    #获取dZ的基本信息
    (m,n_H,n_W,n_C) = dZ.shape

    #获取权值的基本信息
    (f,f,n_C_prev,n_C) = W.shape

    #获取hparameters的值
    pad = hparameters['pad']
    stride = hparameters['stride']

    #初始化各个梯度的结构
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW = np.zeros((f,f,n_C_prev,n_C))
    db = np.zeros((1,1,1,n_C))

    #前向传播中我们使用了pad，反向传播也需要使用，这是为了保证数据结构一致
    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)

    #现在处理数据
    for i in range(m):
        #选择第i个扩充了的数据的样本，降了一维。
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    #定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    #定位完毕，开始切片
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    #切片完毕，计算梯度
                    da_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:] += W[:,:,:,c] * dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]

        #设置第i个样本最终的dA_prev,即把非填充的数据取出来
        dA_prev[i,:,:,:] = da_prev_pad[pad:-pad,pad:-pad,:]

    #数据处理完毕，验证数据格式是否正确
    assert (dA_prev.shape == (m,n_H_prev,n_W_prev,n_C_prev))

    return dA_prev,dW,db
'''
#测试一下
np.random.seed(1)
#初始化参数
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad":2,"stride":1}

#前向传播
Z,cache_conv = conv_forward(A_prev,W,b,hparameters)
#反向传播
dA,dW,db = conv_backward(Z,cache_conv)
print("dA_mean =",np.mean(dA))
print("dW_mean =",np.mean(dW))
print("db_mean =",np.mean(db))
'''

print("----------1.5.2:池化层反向传播------------")
#最大值池化层的反向传播
def creat_mask_from_window(x):
    '''
    从输入矩阵中创建掩码，以保存最大值的矩阵的位置。
    :param x: 一个维度为（f,f）的矩阵
    :return:
            mask：包含x的最大值的位置的矩阵
    '''
    #t = np.max(x)
    mask = x == np.max(x)


    return mask
'''
#测试
np.random.seed(1)

x = np.random.randn(2,3)

mask = creat_mask_from_window(x)

print("x = " + str(x))
print("mask = " + str(mask))
'''


#均值池化层的反向传播
def distribute_value(dz,shape):
    '''
    给定一个值，为按矩阵大小平均分配到每一个矩阵位置中。
    :param dz:输入的实数
    :param shape:元组，两个值，分别为n_H,n_W
    :return:
            a：已经分配好了值的矩阵，里面的值全部一样
    '''
    #获取矩阵的大小
    (n_H,n_W) = shape

    #计算平均值
    average = dz / (n_H*n_W)

    #填充入矩阵
    a = np.ones(shape) * average

    return a
'''
dz = 2
shape = (2,2)
a = distribute_value(dz,shape)
print("a = " + str(a))
'''


#池化层的反向传播封装模型
def pool_backward(dA,cache,mode = "max"):
    '''
    实现池化层的反向传播
    :param dA: 池化层的输出的梯度，和池化层的输出的维度一样
    :param cache:池化层前向传播时所存储的参数
    :param mode:模式选择，【’max‘|’average‘】
    :return:
            dA_prev:池化层的输入的梯度，和A_prev的维度相同
    '''

    #获取cache中的值
    (A_prev,hparameters) = cache

    #获取hparameters的值
    f = hparameters['f']
    stride = hparameters['stride']

    #获取A_prev和dA的基本信息
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (m,n_H,n_W,n_C) = dA.shape

    #初始化输出的结构
    dA_prev = np.zeros_like(A_prev)

    #开始处理数据
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    #定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    #选择反向传播的计算方式
                    if mode == 'max':
                        #开始切片
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]

                        #创建掩码
                        mask = creat_mask_from_window(a_prev_slice)

                        #计算dA_prev
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += np.multiply(mask,dA[i,h,w,c])

                    elif mode == 'average':
                        #获取dA的值
                        da = dA[i,h,w,c]
                        #定义过滤器大小
                        shape = (f,f)
                        #平均分配
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += distribute_value(da,shape)

    assert (dA_prev.shape == A_prev.shape)

    return dA_prev
'''
#测试
np.random.seed(1)
A_prev = np.random.randn(5,5,3,2)
hparameters = {"stride":1,"f":2}
A,cache = pool_forward(A_prev,hparameters)
dA = np.random.randn(5,4,2,2)

dA_prev = pool_backward(dA,cache,mode = "max")
print("mode = max")
print("mean of dA = ",np.mean(dA))
print('dA_prev[1,1] = ',dA_prev[1,1])
print()
dA_prev = pool_backward(dA,cache,mode="average")
print("mode=average")
print("mean of dA = ",np.mean(dA))
print('dA_prev[1,1 = ',dA_prev[1,1])
'''

#神经网络的应用
print("==========Step2:神经网络的应用pytorch===========")

#设置随机种子
torch.manual_seed(1)

#载入数据
X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = load_dataset()

#可视化一个样本
index = 90
plt.imshow(X_train_orig[index])
print("y = " + str(np.squeeze(Y_train_orig[:,index])))
plt.show()

#归一化数据集
X_train = np.transpose(X_train_orig,(0,3,1,2)) / 255     #将维度转化为(1080,3,64,64),,,np.transpose就是转化维度
X_test = np.transpose(X_test_orig,(0,3,1,2)) / 255       #将维度转化为（120，3，64，64）

#转置y
Y_train = Y_train_orig.T                    #(1080,1)
Y_test = Y_test_orig.T                      #(120,1)

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

#创建数据接口
def data_loader(X_train,Y_train,batch_size=64):
    '''
        此函数将X_train,Y_train打包，可以理解为打包成类似于小批量迭代器的形式，每遍历所有批次再进行下一轮迭代，
    它都会自动打乱数据集，使每个批次内的样本保持随机。
    '''
    train_db = TensorDataset(torch.from_numpy(X_train).float(),torch.squeeze(torch.from_numpy(Y_train)))
    # tensordataset每个元素都是张量，功能对tensor打包，将X，Y封装起来
    # shuffle = True,则每次遍历批量后重新打乱顺序
    train_loader = DataLoader(train_db,batch_size=batch_size,shuffle=True)
    # dataloader就是用来包装使用数据，小批量
    return train_loader

#构建模型
class CNN(nn.Module):  #名字叫CNN的类，CNN是子类，nn.Module是父类，继承关系
    #nn.Module是nn中十分重要的类，包含网络各层的定义及前向传播方法
    def __init__(self):#构造函数
        #继承模块
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(  #一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中
            nn.Conv2d(                          #input shape(3,64,64)
                in_channels=3,                  #input通道数
                out_channels=8,                 #output通道数
                kernel_size=4,                  #卷积核的边长f
                stride=1,                       #步长
                padding=1                       #padding模式为SAME，=[(s-1)n-s+f]/2，小数的话，向下取整
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8,stride=8,padding=4)
        )
        self.conv2 = nn.Sequential(          #input shape(8,64,64)
            nn.Conv2d(8,16,2,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4,stride=4,padding=2)
        )
        self.fullconnect = nn.Sequential(
            nn.Linear(16*3*3,20),
            nn.ReLU()
        )
        self.classifier = nn.LogSoftmax(dim=1)#softmax外加log函数

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #展平
        x = x.view(x.size(0),-1)
        x = self.fullconnect(x)
        output = self.classifier(x)
        return output

def weigth_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data,0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data,0)



#封装
def model(X_train,Y_train,X_test,Y_test,learning_rate=0.009,num_epochs=100,minibatch_size=64,print_cost=True,is_plot=True):
    train_loader = data_loader(X_train,Y_train,minibatch_size)
    cnn = CNN()
    #cnn.apply(weigth_init(X_train.shape[0]))
    cost_func = nn.NLLLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),lr=learning_rate,betas=(0.9,0.999))
    #保存每次迭代的cost的列表
    costs = []
    #批次数量
    m = X_train.shape[0]
    num_batch = m / minibatch_size

    for epoch in range(num_epochs):
        epoch_cost = 0
        for step,(batch_x,batch_y) in enumerate(train_loader):
            #前向传播
            output = cnn(batch_x)
            #计算成本
            cost = cost_func(output,batch_y)
            epoch_cost += cost.data.numpy() / num_batch

            #梯度归零
            optimizer.zero_grad()
            #反向传播
            cost.backward()
            #更新参数
            optimizer.step()

        if print_cost and epoch % 5 == 0:
            costs.append(epoch_cost)
            print("Cost after epoch %i:%f" % (epoch,epoch_cost))

    #画曲线
    if is_plot:
        plt.plot(costs)
        plt.xlabel('iterations per 5')
        plt.ylabel("cost")
        plt.show()

    #保存学习后的参数
    torch.save(cnn.state_dict(),'net_params.pkl')
    print('参数已保存到本地pkl文件中。')

    #预测训练集
    cnn.load_state_dict(torch.load('net_params.pkl'))
    output_train = cnn(torch.from_numpy(X_train).float())
    pred_Y_train = torch.max(output_train,dim=1)[1].data.numpy()

    #预测测试集
    output_test = cnn(torch.from_numpy(X_test).float())
    pred_Y_test = torch.max(output_test,dim=1)[1].data.numpy()

    #训练集准确率
    print("训练集准确率：%.2f %%" % float(np.sum(np.squeeze(Y_train) == pred_Y_train) / m * 100))
    #测试集准确率
    print("测试集准确率：%.2f %%" % float(np.sum(np.squeeze(Y_test) == pred_Y_test) / X_test.shape[0] * 100))

    return cnn

model(X_train,Y_train,X_test,Y_test)


