# C++算法与数据结构

## STL

### 容器

#### 1. string

string本质上是一个类，内部还是char*。

### 2. map/multi_map/unorder_map

字典数据结构，本身应该无序，但c++中map和multi-map使用的二叉堆的数据结构，因此内部会对key进行排序，查找时的时间效率是logn，unorder_map则是真正意义上的无序字典。

```
// 增删改查
// 1. 增
map.insert(pair<T, U>(k, v);
map.insert(std::map<T, U>::value_type(k, v);
map[k] = v;

// 2. 查
// 注意c++基本上都是使用迭代器
std::map<T, U>::iterator it = map.begin();
auto it_end = map.end();
for(;it!=it_end;it++) {
   cout << it->first << it->second;
}
auto find_it = map.find(k);
cout << find_it->first << find_it->second;

// 3. 删除

// 4. 修改
map[k] = v;
```

### 3. tuple

## 小技巧

1. 如何将不同的数据放入到容器中。
   - 使用boost库中的variant或者any
   - https://gieseanw.wordpress.com/2017/05/03/a-true-heterogeneous-container/
2. 异质链表
3. 模板元编程（https://www.cnblogs.com/apocelipes/p/11289840.html）
4. 可变参数模板
5. 将函数作为参数传入另一函数。
   其实是传入函数指针，格式如下returnType(星参数名)(函数参数)=函数名

