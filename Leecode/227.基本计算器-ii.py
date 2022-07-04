#
# @lc app=leetcode.cn id=227 lang=python3
#
# [227] 基本计算器 II
#

# @lc code=start
class Solution:
    def calculate(self, s: str) -> int:
        symbol = '+'
        num_res = []
        res = 0
        n = len(s)
        left = 0
        while left < n:
            if s[left] == ' ':
                left += 1
                continue
            if s[left] in ['+', '-', '*', '/']:
                symbol = s[left]
                left += 1
                continue
            i = 0
            while left < n and s[left].isdigit():
                i = 10*i + int(s[left])
                left += 1
            if symbol == '+':
                num_res.append(i)
            elif symbol == '-':
                num_res.append(-i)
            elif symbol == '*':
                num_res.append(num_res.pop()*i)
            elif symbol == '/':
                num_res.append(int(num_res.pop() / i))
            else:
                symbol = s[left]
        for num in num_res:
            res += num
        return res
# print(Solution().calculate(" 3/2 "))
# @lc code=end

