# Docker

一个开源的应用容器引擎，2013年初诞生，使用go语言开发。Docker可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中，然后发布到流行的linux中。

## 1. 安装Docker

```shell
# 自己根据不同的环境安装
# 最后一步查看docker版本
# linux可下载tgz包后自己配置安装，tgz包链接：https://download.docker.com/linux/static/stable/x86_64/，安装指导：https://www.jianshu.com/p/64a470628e49
docker -v 
```

## 2. Docker架构

![img](https://cdn.zsite.com/data/upload/d/docker/202009/f_dec504d3c908d19e6c5165f251d3d124.jpg)

镜像（Image）：Docker镜像，就相当于是一个root文件系统。

容器（Container）：镜像和容器的关系就像是面向对象程序设计中的类和对象一样，镜像是静态的定义，容器是镜像运行时的实体。容器可以被创建、启动、停止、删除、暂停等。

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




## 4. Docker应用部署

### 4.1 MySql部署

需求：在Docker容器中部署Mysql，并通过外部mysql客户端操作Mysql Server。

实现步骤：

1. 搜索mysql镜像
2. 拉取mysql镜像
3. 创建容器
4. 操作容器中的mysql

通信问题：

1. 容器内的网络服务和外部机器不能直接通信。
2. 外部机器和宿主机可以直接通信。
3. 宿主机和容器可以直接通信。
4. 当容器中的网络服务需要被外部机器访问时，可以将容器中提供给服务的端口映射到宿主机的端口上。外部机器访问宿主机的该端口，从而间接访问容器的服务。
5. 这种操作称为：端口映射。

### 4.2 实际部署

1. 搜索mysql镜像

2. 拉取镜像

3. 创建容器，设置端口映射，目录映射

   ```shell
   mkdir ~/mysql
   cd ~/mysql
   ```

   ```shell
   docker run -id \
   -p 3307:3306 \
   --name==c_mysql \
   -v $PWD/conf:/etc/mysql/conf.d \ 
   -v $PWD/logs:/logs \ 
   -v $PWD/data:var/lib/mysql \
   -e MYSQL_ROOT_PASSWORD=123456 \
   mysql:5.6
   ```

   - 参数说明
     - -p 3307:3306：将容器的3306端口映射到宿主机的3307端口。以后基本都是映射一样的端口号。
     - -v $PWD/conf:/etc/mysql/conf.d：将主机当前目录下的conf/my.cnf挂载到容器的/etc/mysql/my/cng。配置目录。
     - -v $PWD/data:/var/lib/mysql：将主机当前目录下的data目录挂载到容器的/var/lib/mysql。数据目录。
     - -e MYSQL_ROOT_PASSWORD=123456：初始化root用户的密码。

   4. 进入容器，操作mysql

      ```shell
      docker exec =it c_mysql /bin/bash
      ```

   5. 使用外部服务器连接。

## 5. Dockerfile

### 5.1 Docker镜像原理

docker镜像本质是什么？：是一个分层文件系统。

docker中一个centos镜像为什么只有200M，而一个centos操作系统的iso文件有好几个G？：centos的iso镜像文件包包含bootfs和rootfs，而docker的centos镜像服用操作系统的bootfs，只有rootfs和其他镜像层。

docker中一个tomcat镜像为什么有500M，而一个tomcat安装包只有70多M？：由于docker中镜像是分层的，tomcat虽然只有70M，但他需要依赖于父镜像和基础镜像，所以对外暴露的tomcat镜像大小500多M。

操作系统组成部分：

- 进程调度子系统
- 进程通信子系统
- 内存管理子系统
- 设备管理子系统
- 文件管理子系统
- 网络通信子系统
- 作业控制子系统

linux文件系统由bootfs和rootfs两部分组成：

- bootfs：包含bootloader（引导加载程序）和kernel（内核）。
- rootfs：root文件系统，包含的就是典型linux系统中的/dev，/prov，/bin，/etc等标准目录和文件。
- 不同的linux发行版，bootfs基本一样，而rootfs不同，如ubuntu，centos等。

docker镜像原理：

- docker镜像是由特殊的文件系统叠加而成。
- 最底端是bootfs，并使用宿主机的bootfs。所以linux装windows就比较困难。
- 第二层是root文件系统rootfs，称为base image。
- 然后再往上可以叠加其他的镜像文件。
- 统一文件系统（Union Files System）技术能够将不同的层整合成一个文件系统，为这些层提供了一个统一的视角，这样就隐藏了多层的村咋子，在用户的角度看来，只存在一个文件系统。
- 一个镜像可以放在另一个镜像的上面。位于上面的镜像称为父镜像，最底部的镜像称为基础镜像。
- 当从一个镜像启动容器时，docker会在最顶层加载一个读写文件系统作为容器。

### 5.2 镜像制作

docker镜像如何制作？

1. 容器转为镜像

   ```shell
   docker commit 容器id 镜像名称:版本号
   docker save -o 压缩文件名称 镜像名称:版本号
   docker load -i 压缩文件名称
   ```

2. dockerfile概念

   - Dockerfile是一个文本文件
   - 包含了一条条的指令
   - 每一条指令构建一层，基于基础镜像，最终构建出一个新的镜像
   - 对于开发人员：可以为开发团队提供一个完全一直的开发文件
   - 对于测试人员：可以直接拿开发时所构建的镜像或者通过dockerfile文件构建一个新的镜像开始工作了。
   - 对于运维人员：在部署时，可以实现应用的无缝移植。

   写法和关键字较多，建议网上查询。

### 5.3 Dockerfile案例

需求：

自定义centos7镜像。要求：

1. 默认登陆路径为/ysr
2. 可以使用vim

实现步骤

1. 定义父镜像：FROM centos:7
2. 定义作者信息：MAINTAINER tao_wei
3. 执行安装vim命令：RUN yum install -y vim 
4. 定义默认的工作目录：WORKDIR /usr
5. 定义容器启动执行的命令：CMD /bin/bash

```shell
docker build -f ./centos_dockerfile -t name:id .
```



