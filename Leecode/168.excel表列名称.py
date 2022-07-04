#
# @lc app=leetcode.cn id=168 lang=python3
#
# [168] Excel表列名称
#

# @lc code=start
class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        res = []
        num = columnNumber
        res_ = ""
        while num:
            _num = num // 26
            if num % 26 == 0:
                num -= 1
                res.append((num % 26) + 1)
                num = num // 26
                continue
            if _num != 0:
                res.append(num % 26)
                num = _num
            else:
                res.append(num % 26)
                break
        for i in res[::-1]:
            res_ += chr(64+i)
        return res_
# @lc code=end

