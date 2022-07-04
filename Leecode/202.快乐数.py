#
# @lc app=leetcode.cn id=202 lang=python3
#
# [202] 快乐数
#

# @lc code=start
class Solution:
    def isHappy(self, n: int) -> bool:
        val = n
        boundary = 2**31-1
        res = set()
        while val != 1 and val < boundary:
            tem = 0
            for i in str(val):
                tem += int(i) ** 2
            val = tem
            if val in res:
                return False
            res.add(val)
        if val == 1:
            return True
        return False

# @lc code=end

