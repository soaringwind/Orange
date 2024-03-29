# Leecode刷题
## 算法
### 1. 双指针

三数之和、

1. 习题解析

   leetcode 16题：[最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/description/)

   ```
   如何从n3降低至n2：外部做一次遍历，内部使用双指针。
   ```

   

### 2. 贪心

课程表排序、最长跳跃距离（55）、最少跳跃次数、加油站（134）、

```
第55题跳跃游戏：只需要注意特殊情况即步长为0的时候。
第45题跳跃游戏ii：需要注意维护实际可达到最远距离与理论能达到的最远距离。
第134题加油站：需要注意1. 怎么判断环，即是否走过一圈；2. 注意当到达某一站点小于0时，说明在那一点之前开始都无法走完，因此最少需要从当前站点开始。

备注：
134题：如何通过索引判断是否走过一圈？
答：1. 通过比较是否超过数组长度，如果超过，则从0开始计数。
2. 直接对数组长度进行取余。
end_index % n，超过则会自动相减。
```

### 3. 位运算

1. 与&

   ```
   100 & 000 = 000
   100 & 100 = 100
   ```

   小技巧：

   - 使用n & (n-1) 可以将最后一位的1变成0，leetcode 231题。
   - 使用n & (-n) 可以直接获取二进制表示的最低位的1。

2. 异或^

   ```
   100 ^ 100 = 000
   100 ^ 000 = 100
   ```

3. 移位：需要注意是在表示范围内，相当于是乘/除2**n。

   ```
   >>: 左移，除2的n次方
   <<: 右移，乘2的n次方
   1 << 2: 1 => 100 => 4
   4 >> 2: 100 => 1 => 1
   ```

4. 正负数的表示方法

   ```
   操作数都是由：原码+补码进行表示。
   补码：第一位为符号位
   正数：补码=原码。
   ```

### 4. 查找（搜索）

1. 线性查找：在容器无序的情况往往只能使用线性查找，虽然效率与容器大小成正比，但不会出错。

   ```python
   class Solution:
       def search(self, nums: List[int], target: int) -> int:
           # 线性查找/二分查找
           for i in range(len(nums)):
               if nums[i] == target:
                   return i
           return -1
   ```

   

2. 二分查找：往往应用在有序的情况下，利用两头的信息来缩小查找范围。

   ```python
   class Solution:
       def search(self, nums: List[int], target: int) -> int:
           left = 0
           right = len(nums) - 1
           while left <= right:
               mid = (left+right) // 2
               if target < nums[mid]:
                   right = mid - 1
               elif nums[mid] < target:
                   left = mid + 1
               else:
                   return mid
           return -1
   ```

   小技巧：

   - 将无序列表先进行排序之后再进行查找。
   - 局部有序的列表也可尝试，如Leetcode 33题。
   
3. 树的遍历：二分查找等经常用于线性数据结构中，而复杂数据结构则需要借助栈/队列来进行遍历。

   1. 先序遍历：当前节点 -> 左子树 -> 右子树。
   2. 中序遍历：左子树 -> 当前节点 -> 右子树。
   3. 后序遍历：左子树 -> 右子树 -> 当前节点。
   复习：树的深度遍历都需要用到栈这个结构，采用递归内部维护了一个栈，不采用递归需要自己手动实现一个栈。
   
   ```python
   node_list = [root]
   while node_list:
      node = node_list.pop() # 这里如果写成pop(0)就是队列因为FIF0
      if node.right:
         node_list.append(node.right)
      if node.left:
         node_list.append(node.left) # 这里之所以要倒着就是因为使用的是栈
   ```

4. 层次遍历：往往应用在和树相关的题目中。

   需要使用的数据结构为队列。
   
   ```python
   node_list = [root]
   while node_list:
      for _ in range(len(node_list): # 这里用for的原因就是可以把一层的节点都访问完。
         node = node_list.pop(0) # 这里就是队列因为FIF0
         if node.left:
            node_list.append(node.left)
         if node.right:
            node_list.append(node.right) # 这里就不用倒着了。
   ```

   如何在每一层遍历完成之后再进入下一层，记住下面这张图：

   ![img](https://labuladong.github.io/algo/images/dijkstra/1.jpeg)

   

5. DFS&BFS：深度优先和广度优先算法，两者往往可以互用解同一类型题目，适用于需要向左、向右等方向进行探索时。

   注意：

   - 不能回头，需要记录自己的路径。
   - 加速方法，使用散列表（set）来记录已经遍历过的点。

6. 习题解析：

   leetcode 34题：[在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

   意义：当列表有新的元素插入时，有两种插入方法，第一，找到不大于这个元素的第一个索引，向前插入；第二，找到不小于这个元素的最后一个索引，向后插入。（倒序插入）

   leetcode 417题：[太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/description/) 

   leetcode 1154题：

   该问题的关键在于计算某个结点的左子树数量及右子树数量。（递归求解）

   

### 5. 排序

1. 冒泡排序：效率最低，但是稳定性很高，时间复杂度n**2。

2. 插入排序：效率略高于冒泡，时间复杂度n**2。

3. 归并排序：属于CBA（比较式算法）中，效率较高的算法，时间复杂度n*logn。

4. 计数排序/桶排序：效率可到达n，但是会消耗一定的空间，时间复杂度可到n。
   
   ```python
   class Solution:
      def merge_sort(self, nums, l, r):
         if l == r:
            return
         mid = (l + r) // 2
         self.merge_sort(nums, l, mid)
         self.merge_sort(nums, mid + 1, r)
         tmp = []
         i, j = l, mid + 1
         while i <= mid or j <= r:
            if i > mid or (j <= r and nums[j] < nums[i]):
               tmp.append(nums[j])
               j += 1
            else:
               tmp.append(nums[i])
               i += 1
         nums[l: r + 1] = tmp

      def sortArray(self, nums: List[int]) -> List[int]:
         self.merge_sort(nums, 0, len(nums) - 1)
         return nums
   ```

5. 堆排序：这个目前没有自己手动实现过，但理论是构建一个最大堆，不断把最顶的元素移到数组右边，不断调整最大堆实现排序。

6. 快速排序：效率最差为n方，理论上是nlogn，思想和归并排序很像，都是分治。注意在确定主元（pivot）的时候，最好要随机选取，才能达到理论的效率。
   
   ```python
   # 1. 范围内随机选取一个值。
   # 2. 将该范围内的值小于主元值移到该值左侧，大于主元值的移到该值右侧。
   # 3. 返回主元值当前索引，并将左右范围内的值递归求解。
   class Solution:
      def randomized_partition(self, nums, l, r):
         pivot = random.randint(l, r)
         nums[pivot], nums[r] = nums[r], nums[pivot]
         i = l - 1
         for j in range(l, r):
            if nums[j] < nums[r]:
                i += 1
                nums[j], nums[i] = nums[i], nums[j]
         i += 1
         nums[i], nums[r] = nums[r], nums[i]
      return i

      def randomized_quicksort(self, nums, l, r):
         if r - l <= 0:
            return
         mid = self.randomized_partition(nums, l, r)
         self.randomized_quicksort(nums, l, mid - 1)
         self.randomized_quicksort(nums, mid + 1, r)

      def sortArray(self, nums: List[int]) -> List[int]:
         self.randomized_quicksort(nums, 0, len(nums) - 1)
         return nums
   ```

7. 习题解析

   leetcode 1122题：[数组的相对排序](https://leetcode-cn.com/problems/relative-sort-array/description/)
   leetcode 912题：[排序数组](https://leetcode.cn/problems/sort-an-array/)

### 6. 动态规划

动态规划的题目比较难以思考，尤其是当维度达到了二维，很难总结出通用的思路。现在发现一维也很难总结出思路，但是一维有一个好处，往往能够相处不用动态规划的土方子解法。

1. 习题解析

   leetcode 1143题：[最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/description/)
   
   leetcode 516题：[最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/description/)
   
   leetcode 152题：[乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)
   ```
   这道题实际上我没有想到动态规划，但是它是一个一维的问题，所以可以用一些别的方法来解（比如，有了前缀乘积数组就一定能够得到任意长度的子数组乘积）。
   动态规划思想：每个位置的最大值来源于两个地方，第一个正数乘上之前的最大值，第二个负数乘上之前的最小值。所以干脆就维护两个数组，一个放最大值，一个放最小值，这样就可以进行动态规划。
   ```

### 7. 回溯

回溯算法往往是用在枚举问题上，有个很大的特点就是每次都是在做相同的事情，因此表现形式基本都是递归。另外，有一点可以发现回溯算法本质上就是一颗二叉树，因此为了提高算法的效率，往往会有很多的剪枝操作。这种题目，最重要的就是先找到最根本的形式，不要先考虑复杂度，否则很容易掉坑，先考虑如何枚举出所有情况，再考虑如何删除不对的情况结果。

1. 剪枝技巧

   - 使用备忘录，记录已经遍历过的情况，不要重复遍历，这种方法实际上就是自顶向下的动态规划。
   - 观察，是否存在不能重叠（N皇后）、找出解之后直接返回等情况。

2. 习题解析：

   leetcode 22题：[括号生成](https://leetcode.cn/problems/generate-parentheses/description/)

   leetcode 17题：[电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/)

   leetcode 51题：[N 皇后](https://leetcode.cn/problems/n-queens/description/)

   leetcode 78题：[子集](https://leetcode-cn.com/problems/subsets/description/)

   ```
   1. 回溯：这题实际上就是一道高中的排列组合题，这类题目均可以考虑使用回溯算法进行求解。大致流程如下：
   def dfs(idx):
   	if 条件成立:
   		res.append(tem)
   	for idx in list:
   		tem.append(list[idx])
   		dfs(idx+1)
   		tem.pop()
   2. 第二种思路，由于是0/1排列，因此可以考虑二进制码来解决。
   ```

   

2. 典型技巧：

   - 记录已经遍历过的点，如果后续再碰到则不再对其进行回溯，以此提高算法效率，避免重复。
   - 当点数很多的时候，记录遍历过的点，使用查询效率为常数的hash表来提高查询效率。

   

## 数据结构

### 1. 链表

链表是一种常见的数据结构，通常是由内存地址作为索引。结构体代码如下。队列可以使用链表来完成，因为其可以在常数时间内完成增加和删除元素。

```python
class ListNode(object):
    # 单向链表
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

1. 检测环：链表中由于是通过索引来连接数据，所以可能会出现环的情况。

   ![image-20220412142210163](C:\Users\weitao\AppData\Roaming\Typora\typora-user-images\image-20220412142210163.png)

   方法：

   - 哈希表：使用哈希表来储存所有的链表节点，如果有重复的，则表示有环。

     好处：简单，查询效率高。

   - 快慢指针：慢节点走一步，快节点走两步，如果两个节点相遇则说明有环。

     好处：效率高，没有查询这一步，且内存消耗小。
   
2. 排序：按照值的大小进行排序，排序算法通用，但是需要进行一定程度的变换。

### 2. 树

1. 遍历

2. 重构：给出一颗二叉树的遍历结果，来反构造回去二叉树的结构，往往是前+中，或者是后+中。特殊情况，满二叉树的情况下，才能由前+后完成。

   - 由有序数组构建二叉搜索树：找到每一个区间的中间位置，递归构建左子树及右子树。

     leetcode 108题：[将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/description/)

 3. 二叉搜索树：该树的中序遍历结果就是一个非降序的结果，也就是说该树的左子树的值一定小于根节点，右子树的值一定大于根节点。


### 3.  散列表

最大的好处在于可以将查询的效率从n降低至1的时间复杂度。散列表在python中最常见的就是字典的数据结构，如果存在n^3降低到n^2的时候，就需要使用到字典。这里注意，两个数使用字典可以从n^2降低到n，因为可以提前储存一个值，而三个数使用字典可以从n^3降低到n^2，因为可以提前储存某两个值的情况。

1. 习题解析

   leetcode 1题：[两数之和](https://leetcode-cn.com/problems/two-sum/description/)
   
   leetcode 982题：[按位与为零的三元组](https://leetcode.cn/problems/triples-with-bitwise-and-equal-to-zero/)

### 4. 栈

一种线性结构，空间与储存的元素相关，即n的空间复杂度，特点是先进后出，所以特别适用于在发生某种情况之后往前处理的情况。

1. 习题解析

   leetcode 394题：[字符串解码](https://leetcode-cn.com/problems/decode-string/description/)

## 特殊问题

### 1. 回文串问题

回文串定义：一个字符串，从左往右和从右往左遍历之后都可以得到同一个字符串数组。

典型列题：

1. 判断一个字符是否为回文串：双指针，左右指针检查元素是否相同，一个增大，一个减小。

2. 找出删除某些字符情况下，可获得的最长回文子串问题：动态规划。

   leetcode 516题：[最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/description/)

### 2. 穷举问题

这类不放回的穷举问题往往可以使用二进制的方法来表示，哪一个元素取到或者没取到。同时还有几种简单的枚举思路，都和二进制类似。

1. 使用集合，由于集合无序因此很适合将其索引放入，不会导致重复枚举。
2. 使用字符串，由于字符串python中不可修改，这里还需要引入数组辅助。
3. 使用二进制，将数字转成二进制，如果下标为1则选取，如果不是则不选取。

### 3. 区间问题

给出多个区间，需要判断是否有区间重复，或者删除重复区间，或者留下不重复的区间等等问题。此类问题，往往是先进行排序，之后再对其进行判断。

小技巧：

1. 两个合理区间不重叠的充要条件：end1 < start2。
2. 判断区间是否合理：start1 <= end1。

## 题目
动态规划：53，1143，121，122，516

## 注意事项

1. python取整是向下取整，而对于负数是向负无穷取整，因此-1.2会变成-2，所以需要使用int+除数，来达到取整效果。
2. python全局变量与局部变量的定义差别。
