#
# @lc app=leetcode.cn id=20 lang=python3
#
# [20] 有效的括号
#

# @lc code=start
class Solution:
    def isValid(self, s: str) -> bool:
        res = []
        for i in range(len(s)):
            if s[i] == ')' and res:
                item = res.pop()
                if item != '(':
                    return False
            elif s[i] == '}' and res:
                item = res.pop()
                if item != '{':
                    return False
            elif s[i] == ']' and res:
                item = res.pop()
                if item != '[':
                    return False
            else:
                res.append(s[i])
        if res:
            return False
        else:
            return True
print(Solution().isValid(r"()[]{}"))
# @lc code=end

