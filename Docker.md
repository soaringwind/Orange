# Docker

一个开源的应用容器引擎，2013年初诞生，使用go语言开发。Docker可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中，然后发布到流行的linux中。

## 1. 安装Docker

```shell
# 自己根据不同的环境安装
# 最后一步查看docker版本
docker -v 
```

## 2. Docker架构

![img](https://cdn.zsite.com/data/upload/d/docker/202009/f_dec504d3c908d19e6c5165f251d3d124.jpg)

镜像（Image）：Docker镜像，就相当于是一个root文件系统。

容器（Container）：镜像和容器的关系就像是面向对象程序设计中的累和对象一样，镜像是静态的定义，容器时镜像运行时的实体。容器可以被创建、启动、停止、删除、暂停等。

仓库（Repository）：仓库可看成是一个代码控制中心，用来保存镜像。

默认从docker hub上下载，比较慢，如果有必要则可以配置镜像加速器。

## 3. Docker命令

### 3.1 Docker服务相关命令

```shell
# 启动docker服务
systemctl start docker 
# 查看docker服务状态
systemctl status docker 
# 停止docker服务状态
systemctl stop docker 
# 重启docker服务状态
systemctl restart docker
# 开机自启动docker服务状态
systemctl enable docker
```

### 3.2 Docker镜像相关命令

```shell
# 查看镜像
docker images 
docker images -q # 查看所有镜像的id
# 搜索镜像
docker search redis 
# 下载镜像
docker pull redis
# 查看镜像的版本(上hub.docker.com查看版本)
docker pull redis:5.0
# 删除镜像
docker rmi IMAGE ID
docker rmi redis:latest 
```

### 3.3 Docker容器相关命令

```shell
# 查看容器
docker ps -a # 所有
# 创建容器 -i: 保持容器无论是否有客户端连接都一直运行 -id: 以守护（后台）模式运行容器，容器退出也不会关闭 -it：容器创建后自动进入，退出容器关闭 --name取名字
docker run -i -t --name=test centos:7 /bin/bash
docker run -id --name=test1 centos:7
# 进入容器
docker exec -it test1 /bin/bash
# 退出容器
exit
# 启动容器
docker start test1
# 停止容器
docker stop test1 
# 删除容器(不能删除正在运行的容器)
docker rm test1 
# 查看容器信息
docker inspect test1 
```

### 3.4 Docker容器的数据卷

#### 3.4.1 数据卷的概念和作用

数据卷：数据卷是宿主机中的一个目录或文件。当容器目录和数据卷目录绑定后，对方的修改会立即同步。一个数据卷可以被多个容器同时挂在。一个容器也可以被挂载多个数据卷。

数据卷作用：

- 容器数据持久化
- 外部机器和容器间接通信
- 容器之间数据交换

数据卷容器：

- 创建一个容器，挂载一个目录，让其他容器继承自该容器(--volume-from)。
- 通过简单方式实现数据卷。

配置数据卷：

- 创建启动容器时，使用-v参数设置数据卷

  ```shell
  docker run ... -v 宿主机目录（文件）：容器内目录（文件） ...
  ```

- 注意事项：
  1. 目录必须是绝对路径
  2. 如果目录不存在，会自动创建
  3. 可以挂载多个数据卷（一直输入-v即可
  4. 不同容器可以挂载同一个数据卷

##### 数据卷容器

多容器进行数据交换：

1. 多个容器挂载同一个数据卷
2. 数据卷容器

配置数据卷容器

1. 创建启动c3数据卷容器，使用-v参数设置数据卷

   ```shell
   docker run -it --name=c3 -v /volume centos:7 /bin/bash
   ```

2. 创建启动c1, c2容器，使用--volumes-from参数设置数据卷

   ```shell
   docker run -it --name=c1 --volumes-from c3 centos:7 /bin/bash
   docker run -it --name=c2 --volumes-from c3 centos:7 /bin/bash
   ```

   