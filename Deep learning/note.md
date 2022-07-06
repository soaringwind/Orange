### 深度可分离卷积
深度可分离卷积depthwise separable convolution，由depthwise(DW)和pointwise(PW)两个部分结合起来，用来提取特征feature map相比常规的卷积操作，其参数数量和运算成本比较低。实际上就是把一次卷积做的事情，拆成两次，来降低参数量。
常规卷积操作

![image](https://user-images.githubusercontent.com/26198992/177484110-729ff874-fc3e-453b-b5dd-55380e75821f.png)

深度可分离卷积操作
1. 逐通道卷积

![image](https://user-images.githubusercontent.com/26198992/177484224-b9ced2b4-3c27-4117-a92e-2b4c2d16149f.png)

2. 逐点卷积

![image](https://user-images.githubusercontent.com/26198992/177484263-e3fa3192-32e5-4a7f-a5cd-600aa9db6cef.png)

通过计算可以发现，参数的数量明显减少。
