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

### 排序
sorted的用法我是没想到还能这么用。

119.3.238.207