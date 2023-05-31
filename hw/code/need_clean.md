# 可信考试

## 编程题

### 双指针

1. 1794. 【软件认证】最长的指定瑕疵度的元音子串(http://oj.rnd.huawei.com/problems/1794/submissions)
双指针思路，类似于滑动窗口，但是需要先确定右边的位置，再移动左边界的位置。如果先确定左边界位置，则较为繁琐。也是可以做的。
```
class Solution:
    def longest_flawedvowel_substr_len(self, flaw, input_string):
        # 在此添加你的代码
        left = 0 
        max_len = 0  
        count = 0
        check = set([i for i in "aeiouAEIOU"])
        for right in range(len(input_string)):
            if input_string[right] not in check:
                count += 1
                continue 
            if count < flaw:
                continue 
            if count == flaw and input_string[left] in check:
                max_len = max(max_len, right-left+1)
                continue  
            while count > flaw or input_string[left] not in check:
                left += 1
                if input_string[left-1] not in check:
                    count -= 1
                if left > right:
                    left = right
                    break 
            if count == flaw and input_string[left] in check:
                max_len = max(max_len, right-left+1)
        return max_len
```
注意滑动窗口的时候一定要先动右指针，因为要先扩大窗口再缩小窗口，如果先动左指针就很复杂了。

### 排序
sorted的用法我是没想到还能这么用。sorted非常常用，一定要注意，可以使用key这个关键来进行排序。
```
sorted(x_list, key=lambda x: x[1]...) # 这里的key后面是可以写for循环的。
```

### 哈希函数
https://blog.csdn.net/qq_41603898/article/details/85162195

### 完全二叉树和满二叉树的区别
https://blog.csdn.net/mawming/article/details/46471429
