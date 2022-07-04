#
# @lc app=leetcode.cn id=17 lang=python3
#
# [17] 电话号码的字母组合
#

# @lc code=start
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        digits_str = {
            "2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"
        }
        res = ""
        res_list = []
        n = len(digits)
        def dfs(digits_index, res):
            if digits_index >= n:
                res_list.append(res)
                return
            for i in digits_str[digits[digits_index]]:
                res += i
                dfs(digits_index+1, res)
                res = res[:-1]
            return
        dfs(0, res)
        return res_list

# @lc code=end

