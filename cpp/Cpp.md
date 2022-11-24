# C++算法与数据结构

## STL

### 容器

#### 1. string

string本质上是一个类，内部还是char*。

#### 2. map/multi_map/unorder_map

字典数据结构，本身应该无序，但c++中map和multi-map使用的二叉堆的数据结构，因此内部会对key进行排序，查找时的时间效率是logn，unorder_map则是真正意义上的无序字典。

```c++
// 增删改查
// 1. 增
map.insert(pair<T, U>(k, v));
map.insert(std::map<T, U>::value_type(k, v));
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
map.erase(iterator)

// 4. 修改
map[k] = v;
```

#### 3. tuple

#### 4. stack
这个实际上就是容器适配器，类似于栈的数据结构。注意pop方法是没有返回值的，如果想拿到栈顶元素，需要使用top方法。

#### 5. vector
vector容器是c++中可变长的数组，一般情况下使用vector较多，如果没有特别情况，使用它即可。

#### 6. deque
deque是双端链表，因此其头插和尾插的效率很高，头删和尾删的效率也很高，但查找效率很低。

#### 7. set
set实际上是一个集合，里面储存的元素必须不同，且初学者尽量不要尝试修改里面的值。且内部是做了排序。如果想要存放自定义的数据结构，则必须要重载()运算符。
```c++
#include <set>

// 创建set集合
std::set<T, cmp> set;

// 增
set.insert(T);

// 查
set.find(T);

// 删
set.erase(T);

// 遍历
for (auto it=set.begin(); it!=set.end(); it++) {
   cout << *it << endl;
}
```

#### queue
c++中的队列，priority_queue为大顶堆，也就是大的数放在了前面，如果想要小顶堆则需要重载比较方法。

```c++
#include <queue>

// 初始化
priority_queue<T> q;
priority_queue<T, vector<T>, cmp> q;
```


## 小技巧

1. 如何将不同的数据放入到容器中。
   - 使用boost库中的variant或者any
   - https://gieseanw.wordpress.com/2017/05/03/a-true-heterogeneous-container/
2. 异质链表
3. 模板元编程（https://www.cnblogs.com/apocelipes/p/11289840.html）
4. 可变参数模板
5. 将函数作为参数传入另一函数。
   其实是传入函数指针，格式如下returnType(星参数名)(函数参数)=函数名
6. cout输出的时候默认有效位数是6位，因此对于double等数据类型输出不准确，如果需要准确输出则需要考虑调整有效位数。

