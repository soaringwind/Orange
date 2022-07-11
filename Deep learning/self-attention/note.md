# 深度可分离卷积
深度可分离卷积depthwise separable convolution，由depthwise(DW)和pointwise(PW)两个部分结合起来，用来提取特征feature map相比常规的卷积操作，其参数数量和运算成本比较低。实际上就是把一次卷积做的事情，拆成两次，来降低参数量。
常规卷积操作如下图，输入的channel数量为3，想要得到4个channel的feature map，那需要的模板的数量实际上是12个3×3的模板，有108个参数需要学习，参数实际上是相对较多的。

![image](https://user-images.githubusercontent.com/26198992/177484110-729ff874-fc3e-453b-b5dd-55380e75821f.png)

深度可分离卷积操作，由两个部分组成，首先是逐通道卷积，也就是输入和输出的通道数是相同的，这时候需要的参数量是27个，之后进行逐点卷积，这个使用的是1×1的模板来调整通道数，输入3通道，输出的是4通道，因此需要的参数量是12个，加在一起也不过39个，参数量远远低于之前的常规卷积操作。
1. 逐通道卷积

![image](https://user-images.githubusercontent.com/26198992/177484224-b9ced2b4-3c27-4117-a92e-2b4c2d16149f.png)

2. 逐点卷积

![image](https://user-images.githubusercontent.com/26198992/177484263-e3fa3192-32e5-4a7f-a5cd-600aa9db6cef.png)

通过计算可以发现，参数的数量明显减少。

# ViT
在提取的特征矩阵之前，加上一个cls_token的标志，该位对应向量可以作为整句话的语义表示，从而用于下游的分类任务等。这是因为cls_token位本身没有语义，经过多层attention，得到的是attention后所有词的加权平均，相比其他正常词，可以更好的表征句子语义。

# self-attention原理
自注意力机制（self-attention）最开始用在NLP任务中，它可以有效的解决RNN和LSTM带来的记忆消退问题。RNN（循环神经网络）它最大的问题在于两点，第一是无法并行计算，只能够等待前一个计算完之后，再计算下一个输出，第二是无法将最前面的信息跟随着传递到最后面去（也就是记忆消退问题）。在此基础上提出attention机制，它首先是考虑全局的影响，之后根据训练来找到需要注意的地方，也就是自适应的找到聚焦点。如下图。

![image](https://user-images.githubusercontent.com/26198992/177941193-149f8549-1bd1-48e3-abf8-0c280d40de16.png)

## attention原理
注意力机制的提出要早于自注意力机制，它实际上解决的问题是如何分配注意力问题，也是去计算attention值基本和后面的self-attention的计算过程一样。本质过程如图。

![image](https://user-images.githubusercontent.com/26198992/177942431-8420f8fc-ed7d-4ffe-b0fc-6f2a0b7ac676.png)

query和key计算相似度，之后将其归一化，归一化之后再和value值加权求和，即可得到attention的值。计算如图。

![image](https://user-images.githubusercontent.com/26198992/177942753-98fca16c-0351-4515-b13c-82e41186b4a4.png)

## self-attention原理
self-attention和attention机制在本质上基本一样，他们的区别在于attention是target和source做attention计算，而self-attention是source和source本身计算attention也就是计算自己各个部分的attention分数。计算过程如图。

![image](https://user-images.githubusercontent.com/26198992/177944172-f3a352f6-555b-48ef-b1fd-7903915b2238.png)

# Transformer

## end-to-end

end-to-eng实际上就是端到端，它指的就是输入的是原始数据，输出的是最后的结果。而在最初的机器学习当中，输入的往往是在原始数据中提取的特征，这种时候分类的结果十分取决于提取特征的好坏，所以以前的机器学习又被称为特征工程（feature engineering）。

好处：通过缩减人工预处理和后续处理，尽可能使模型从原始输入到最终输出，给模型更多可以根据数据自动调节的空间，增加模型的整体契合度。

缺点：往往需要大量的训练数据。比如人脸识别，无法提前知道人脸会在何处出现，也不知道大小是多少，很难直接从原始图像中直接判断，这时候就需要分步来完成。

## transformer

![img](https://upload-images.jianshu.io/upload_images/1667471-926eb6cb29978dad.png?imageMogr2/auto-orient/strip|imageView2/2/w/347/format/webp)



