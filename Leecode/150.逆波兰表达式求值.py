#
# @lc app=leetcode.cn id=150 lang=python3
#
# [150] 逆波兰表达式求值
#

# @lc code=start
from typing import List


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        res = []
        for i in tokens:
            if i not in ['+', '-', '*', '/']:
                res.append(int(i))
            else:
                b = res.pop()
                a = res.pop()
                if i == '+':
                    res.append(a+b)
                elif i == '-':
                    res.append(a-b)
                elif i == '*':
                    res.append(a*b)
                else:
                    res.append(int(a/b))
        return res[0]

print(Solution().evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"]))
# @lc code=end

