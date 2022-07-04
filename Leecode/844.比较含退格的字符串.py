#
# @lc app=leetcode.cn id=844 lang=python3
#
# [844] 比较含退格的字符串
#

# @lc code=start
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        res_1 = []
        res_2 = []
        for i in s:
            if i == "#":
                if not res_1:
                    continue
                res_1.pop()
            else:
                res_1.append(i)
        for i in t:
            if i == "#":
                if not res_2:
                    continue
                res_2.pop()
            else:
                res_2.append(i)
        return res_1 == res_2
# @lc code=end

