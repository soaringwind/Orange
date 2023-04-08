# Vue+Flask开发

## 1. 安装nodejs

### 1.1 windows安装nodejs

参考：https://blog.csdn.net/Small_Yogurt/article/details/104968169

### 1.2 linux安装nodejs

在官网下载之后还需要进行软连接的配置，需要把解压出来的文件夹下的bin文件路径配置到/usr/local/bin/npm等里面去，否则无法在别的地方使用。

## 2. nodejs安装vue
使用npm install -g @vue/cli命令安装之后，同样需要配置软连接，否则无法在其他地方使用。

## 2.1 vue创建项目
1. 输入vue create test创建一个新的vue项目。
2. 采取手动创建，选择多个特性，router，vuex，css等。其余可随意选择，等待创建即可。
3. 进入文件夹，输入vue add element-plus命令。
4. 输入npm install axios安装axios插件。
5. 在项目中引入axios并配置http。

### 2.1.1 配置axios的http转发
前后端的交互就是通过请求来发起，因此需要对其进行axios的配置，主要是需要配置基础路由和请求配置，响应配置等等。这个的配置也很简单，就是在http的index.js中配置即可。

```js
import axios from 'axios'

// axios create 创建一个axios实例 给实例编写配置，后续所有通过实例发送的请求，都受当前配置的约束
const $http = axios.create({
    baseURL: '', 
    timeout: 1000,
}); 

// 添加请求拦截器
$http.interceptors.request.use(function (config) {
    // 在发送请求之前做些什么
    config.headers.token = '11212' 
    return config;
  }, function (error) {
    // 对请求错误做些什么
    return Promise.reject(error);
  });

// 添加响应拦截器
$http.interceptors.response.use(function (response) {
    // 2xx 范围内的状态码都会触发该函数。
    // 对响应数据做点什么
    let data = response.data 

    return data;
  }, function (error) {
    // 超出 2xx 范围的状态码都会触发该函数。
    // 对响应错误做点什么
    return Promise.reject(error);
  });
  export default $http
```

### 2.1.2 配置子路由
子路由的场景通常出现在希望在某一个主页面下，根据不同的选项而展示不同的辅页面。配置也很简单，就是在router的index.js中在主页面路由下配置children选项。
```js
const routes = [
  {
    path: '/Home',
    name: 'Home',
    component: HomeView, 
    meta: {
      isshow: false,
    }, 
    children: [
      {
        path: '/CourseList', 
        name: 'CourseList', 
        component: () => import('../views/CourseView.vue'), 
        meta: {
          isshow: true, 
          title: "课程列表"
        }
      }, 
    ]
  },
  {
    path: '/',
    name: 'Login',
    component: LoginView, 
    meta: {
      isshow: false,
    }
  }, 
]
```

## vscode配置remote访问远程服务器问题

1. 本地最好使用git的ssh程序。
2. 本地需要生成id_rsa私钥和公钥。
3. 需要注意远程服务器中.ssh文件及文件中的权限有固定要求不可随便更改。

## 解决System limit for number of file watchers reached

sysctl fs.inotify.max_user_watches=524288 
sysctl -p

## 使用python对jpg图片转ico
```python
from PIL import Image

img = Image.open(r"E:\ProgramData\conda_workspace\study\系统安全实验\skey\forms\钥匙.png")
# icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
icon_sizes = [(64, 64)]
img.save('logo.ico', sizes=icon_sizes)
```
































百度地图添加自定义的html标签内容
使用的是control方法，需要先设置好control，之后把html的内容全部放到里面，最后在map中加入进去。
        function SelectControl(){
            this.defaultAnchor = BMAP_ANCHOR_TOP_LEFT; 
            this.defaultOffset = new BMap.Size(10, 10);
        }
        SelectControl.prototype = new BMap.Control(); 
        SelectControl.prototype.initialize = function(map){
            var div = document.createElement("div"); 
            var sel = document.createElement("select");
            var but = document.createElement("button");
            var total_opt = document.createElement("option");
            sel.id = "test";
            but.innerText = "确定";
            but.type = "submit";
            but.name = "formbtn"; 
            but.onclick = showSelectHtml;
            total_opt.value = "total";
            total_opt.text = "total";
            sel.add(total_opt, null);
            for (key in info_dict) {
                opt = document.createElement("option");
                opt.value = key;
                opt.text = key;
                sel.add(opt, null);
            }
            div.append(sel);
            div.append(but);
            map.getContainer().appendChild(div);
            return div; 
        }
        var myZoomCtrl = new SelectControl(); 
        map.addControl(myZoomCtrl);
        function showSelectHtml() {
            console.log($("#test option:selected").text());
        }
