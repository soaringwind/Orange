#
# @lc app=leetcode.cn id=22 lang=python3
#
# [22] 括号生成
#

# @lc code=start
from typing import List


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res_list = []
        res = ""
        def is_valid(brackets):
            tem = []
            for word in brackets:
                if word == "(":
                    tem.append(word)
                elif word == ")" and tem:
                    tem.pop()
                else:
                    return False
            return True
        def dfs(res, left_num):
            if left_num > n or len(res)-left_num > n or left_num < len(res) - left_num:
                return
            if left_num == n and len(res) == 2*n and is_valid(res):
                res_list.append(res)
                return
            for i in "()":
                res += i
                if i == "(":
                    left_num += 1
                dfs(res, left_num)
                if i == "(":
                    left_num -= 1
                res = res[:-1]
            return
        dfs(res, 0)
        return res_list
# print(Solution().generateParenthesis(3))
# @lc code=end

