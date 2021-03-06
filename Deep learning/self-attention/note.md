# 模型度量标准

## 混淆矩阵

<img src="https://img2018.cnblogs.com/common/721540/202002/721540-20200227190850638-263689169.png" alt="img" style="zoom: 80%;" />

TP指的是正样本识别为正样本，TN指的是负样本识别为负样本。实际中，希望TP和TN越多越好，但结果往往不是，所以需要用各种指标来衡量系统的好坏。

![img](https://ask.qcloudimg.com/http-save/1692602/nek3uj4izl.png?imageView2/2/w/1620)

准确率：正确分类样本与所有样本的比率，这个指标是最为严格的。正样本和负样本都需要正确分类，个人比较喜欢用。

精确率/查准率（Precision）：模型预测是正样本中模型预测正确的比率。只考虑模型预测中正样本正确分类比率，负样本考虑较少，类似于挑西瓜中就想要挑到好的。只想知道我挑到的好的里面有多少是好的。适用的场景：如果错误样本被预测为正样本的代价很高，就需要用这个比率，比如垃圾邮件分类，把非垃圾邮件归到垃圾邮件代价就很高。

召回率/查全率（Recall）：在所有正样本中被正确预测的比率。也就是原本是正样本，预测也是正样本占所有正样本比率。也就是挑西瓜中，本来是好西瓜，挑到好西瓜是好西瓜的比率。适用场景：如果正样本被预测为错误的代价很高，就需要用这个比率，比如新冠疫情，如果患病者被预测为未患病者就很糟糕，就需要用这个比率。

特异度：在所有负样本中被正确预测的比率。

## 多标签分类损失
Hamming loss：它统计了误分类标签（这里的误分类统计，统计的是错误的标签个数，比如一个样本是100，预测为000，则损失为1/3）的个数占整个数据的比例。它有个缺陷，如果每个样本本身的标签很稀疏，那么即使每个样本全部预测为0，该损失也会很小。
https://www.zhihu.com/question/358811772


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
在提取的特征矩阵之前，加上一个cls_token的标志，该位对应向量可以作为整句话的语义表示，从而用于下游的分类任务等。这是因为cls_token位本身没有语义，经过多层attention，得到的是attention后所有词的加权平均，相比其他正常词，可以更好的表征句子语义。ViT的关键就是如何把图像使用Transformer来进行分类，如果直接把图像完全展开，直接的问题就是计算量太大了。还有一点是，ViT的表现如果想要超过CNN，那需要拥有足够多的数据进行预训练。另外，ViT只使用了Transformer中的Encoder的部分，下面介绍ViT的流程。

![img](https://pic4.zhimg.com/80/v2-5afd38bd10b279f3a572b13cda399233_1440w.jpg)

ViT流程：

1. Patch embedding：对一张图进行裁剪，ViT使用的是卷积，SwinTransformer则是用unfold来裁剪。比如图片大小为224×224，用步长为16，窗口为16的卷积核，就可以把图像分为14×14个patch，每个patch的特征向量是16×16×3长度，因此输入序列的长度就是196，而每个序列的特征向量长度就是768。这里还要加入一个之前说到的分类向量cls，因此最终的维度是197×768。这就把视觉问题转化成了一个seq2seq问题。
2. Positional encoding：由于图像不同的位置同样可能存在信息，所以还需要有位置编码信息。这里是把位置编码直接加到之前的embedding结果上，并且位置编码让机器自己去学习。
3. LN/multi-head attention/LN：LN输出维度不会变仍旧是197×768。多头自注意力时，先把输入映射到q，k，v，如果只有一个头，那么qkv的维度都是197×768，如果有12个头（768/12=64），则qkv的维度是197×64，一共有12组qkv，最后再将12组qkv的输出拼接起来，输出维度就是197×768，然后再经过一层LN。
4. MLP：将维度放大再缩小回去，197×3072放大为3072，再缩小变为197×768。

这里看出，经过一个block的输出和最开始的输入是一样的维度，因此可以堆叠多个block，最后将第一维的cls作为输出，后面接一个MLP进行图片分类。

这里拿cifar10的数据集使用ViT来分类的时候，有两个问题，第一个训练时间过长，第二个模型容易过拟合，这两个问题都是由于模型过于复杂，往往训练到最后损失已经很小，但是在验证集上的效果并不好。这个问题在网上查过，主要有两个方法，第一个是增加训练的数据量，还没有尝试过，第二个是降低模型复杂度，比如添加dropout，这个尝试过，但是带来一个新的问题，模型收敛速度变慢，第三个办法和第一个类似先在大数据集上训练，再在小的数据集上进行微调。

# Swin Transformer

先说下，这个模型尽管说起来很强大，但是本质上还是把Transformer用在分类上，因此基本流程还是很像，还有self attention的过程。

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


## Swin Transformer
# 代码
https://github.com/berniwal/swin-transformer-pytorch
https://zhuanlan.zhihu.com/p/542675669
https://zhuanlan.zhihu.com/p/361366090

# mask部分理解
循环移位是为了让图像分割方便，因为移位之后，可以和之前一样来进行分割。
实际上这里的mask是因为在之前做了一次循环移位，也就是把较远位置的图像移到了一起，所以互相之间不能计算相关性，要把结果给隐去，所以使用了mask。
https://itcn.blog/p/0856239139.html

# einsum（爱因斯坦求和）
https://zhuanlan.zhihu.com/p/44954540

# 整理
https://www.cxybb.com/article/weixin_44485421/119425070
https://zhuanlan.zhihu.com/p/384727327

# 目标检测
知识点：
1. IOU计算：两个区域交集和并集的比例。https://zhuanlan.zhihu.com/p/111013447
2. 人脸数据集与非人脸数据集准备：在人脸附近随机取区域，把取到的区域和人脸区域进行IOU计算，小于一定比例的认为是非人脸。
3. 级联思想：不会一次就判断该区域是否为人脸，而是先删去大量不是人脸的区域，再层层严格的确定一定为人脸的区域。
4. 非极大值抑制：需要在一个区域内多个重叠框输出为一个框。

## Adaboost
设计出了一个很简单的检测人脸的模板特征（Harr特征），该特征由5个局部特征组合而成，通过检测这五个不同大小的局部特征，可以准确判断出人脸。为了加快计算Harr特征，提出了积分图的概念，有了积分图，计算Harr特征就仅仅只需要几次加减法即可。有了Harr特征之后，它用到了级联的思想也就是Adaboost，通过构建决策树模型，构建出多个强分类器，不断把非人脸剔除，最终留下人脸。

## MTCNN
设计出了三个网络用来进行人脸的检测。其实整体流程和前面的Adaboost非常相似，但是这里用的是神经网络的方式来进行检测，并且用了三个网络，形式上类似级联的思想。
知识点：
1. 提出三个网络，Pnet，Rnet和Onet。每个网络检测图像的尺寸不同，从大到小为12，24和48。每个神经网络都不是很复杂，因此很容易收敛。
2. 使用级联的思想，把复杂问题简单化，因此对设备的要求不高。
3. 检测的时候不断缩小图像的尺寸，构建出图像金字塔，通过第一个Pnet网络可以过滤掉大量不是人脸的区域。之后根据Pnet网络的结果去拿到相应的区域，再输入到Rnet中，再过滤掉大量非人脸区域，再取到对应的区域，再输入到Onet网络中，进行最后的判断。
https://blog.csdn.net/ssunshining/article/details/108903871

# 目标识别
目标识别任务是给出一组特征数据，需要给出该数据所对应的标签。这里CV中最常用的就是卷积神经网络。在实际任务中用到的是一下几个网络。

## VGG
通过组合多个小的卷积核来得到和一个大的卷积核一样的感受野，现在常常用来做特征的提取。因为使用传统的VGG有很多问题，比如直筒式模型找到的特征有限，计算速度较慢。但是在我们的任务中，VGG网络已经表现的足够好。
知识点：
1. 感受野大小：VGG使用的全部是尺寸为3的卷积核，通过组合多个小的卷积核来得到和一个大的卷积核一样大小的感受野。2个3的卷积核和5的卷积核感受野相同，3个3的卷积核和7的卷积核感受野相同。
2. 使用尺寸为2的最大池化来减小窗口的大小，并且随着窗口尺寸的减半，通道数加倍。
3. 引入了全局平均池化的概念，也就是把最后得到的特征图根据通道数直接平均得到一个和通道数相同的向量。

## SENet
在之前的卷积中不断的扩大通道数的目的是为了检测到更多的特征，而每次往下传递的时候，每个通道也就是特征的权重都是相同的，如果每次能够把重要的特征权重提高，不重要的特征权重下降，理论上可以提高模型性能。SeNet就是做这个事情，它分为sequeeze压缩特征块，和excitation激发特征块组成。其实网络的构建非常简单，全局平均池化得到每个特征的值，再接上两层全连接层，最后输出每个通道的权重值。
知识点：
1. SeNet可以很好的嵌入到各个网络中去。
2. 它的思想类似于计算注意力，注意力也是去计算一个权重。后面的CBAM思想很类似。
https://zhuanlan.zhihu.com/p/32702350

## CBAMNet
CBAM的思想和前面的SeNet网络其实是很相似的，它提出一种简单有效的注意力模块。它主要提出两个方面，第一个是通道间的注意力模块（CAM），第二个是空间注意力模块（SAM）。其中CAM和之前的SeNet很相似，首先对每一个特征进行全局池化（最大和平均）之后进入MLP得到通道间注意力，SAM则是对特征图上每一个像素，基于channel进行池化（最大和平均），最后得到两个特征图，再进行尺寸为7的卷积操作，降维成1个通道，最后经过sigmoid得到空间注意力。
知识点：
1. 两种注意力机制计算。
2. 引入注意力机制的确让神经网络更关注我们认为重要的位置。
3. 同样，它也可以融入到各个网络中去。

## SwinT
之前的方法基本上都是在用卷积处理，也用到了很多的注意力机制，但注意力机制提到最多的还是在语音识别任务中，也就是去计算不同tokens之间的相关性，因此有了ViT。ViT将图片进行分割，使得识别任务也能够使用注意力机制来计算。ViT采取的做法是将图片分割为n个尺寸为16的图片块，接下来将每个图片块映射为768维token，这样就把计算机视觉任务改成了一个语义识别任务。但是实作的时候发现，使用这种方法必须先进行大量的预运算，才能够得到超过CNN的结果，如果直接训练，则效果不如CNN，并且该网络需要很长时间的训练，不易收敛。
尽管ViT成功的将计算机识别任务转换成了语义识别任务，但是还是存在很多，比如模型过于复杂，在小数据集上表现一般，且该方法无法得到图像多尺度的信息。SwinT和ViT类似，都是将图像识别任务转换成了语义识别任务，但是ViT更像是硬套，而SwinT则有更多的图像本身信息。SwinT会对图像进行多次的降采样，因此可以得到图像多尺度的信息。其次，SwinT进行的注意力计算，是集中在每个局部的小窗口上，因此注意力的运算量大大减小。同时为了得到全局的注意力信息，会对窗口进行一次滑动，滑动之后再进行一次注意力的计算，以此得到全局的注意力信息。

## FETrans
这个网络模型是中移提出，来解决干扰单标签及多标签的分类任务。如果仅仅是单标签，使用VGG或者GoogLeNet可以得到比较的结果，而如果是多标签分类，单用VGG则效果不是那么优良，如果外场实际使用则效果还需要提升，因此提出了很多方法，比如前面的SeNet，CBAM等等。同时，还尝试使用Transformer来进行，实验表明，如果单单用Transformer效果并不好，且收敛时间很长。因此，提出了一种网络，首先对数据进行卷积，得到更有特征的干扰数据，之后再将数据进行堆叠，之后使用Transformer来进行分类。并且，不是完全的硬套Transformer，位置编码等信息，我们这里并不需要，因此都可以删除。在大量数据的训练下，多标签分类任务，FETrans模型效果会更好一些。

## 常见问题
1. 感受野计算
2. IOU计算
3. 深浅copy
4. 多进程和多线程的区别
