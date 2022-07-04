# 机器学习

## 1. SVM（支持向量机）

SVM(Support vertor machine)支持向量机是用来解决分类问题，尤其是典型的二分类问题如下图。该图的目标就是找到一条线能够将叉和圆分开，如果是多维的数据，就是找到一个超平面来分割数据。SVM就是解决如何找到一个最好的超平面来分割数据。

![这里写图片描述](https://img-blog.csdn.net/20140829135959290)

假设有一超平面可以将数据划分，则该超平面可以写成：

<img src="C:\Users\weitao\AppData\Roaming\Typora\typora-user-images\image-20220601004715272.png" alt="image-20220601004715272"  />

如果该超平面能够将训练的样本进行正确分类，则对于任何训练样本都能得到如下判断：

![image-20220601004925041](C:\Users\weitao\AppData\Roaming\Typora\typora-user-images\image-20220601004925041.png)

如果该超平面能够将训练样本正确分类，则正样本一定在该平面上方，因此肯定大于0，再结合缩放变换，则可以将其变为大于+1，负样本同理，因此可得到上图结果。