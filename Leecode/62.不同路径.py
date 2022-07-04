#
# @lc app=leetcode.cn id=62 lang=python3
#
# [62] 不同路径
#

# @lc code=start
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        a = 1
        for i in range(1, m+n-1):
            a *= i
        b = 1
        for i in range(1, m):
            b *= i
        c = 1
        for i in range(1, n):
            c *= i
        return a // (b*c)
# @lc code=end

