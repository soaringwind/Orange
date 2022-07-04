#
# @lc app=leetcode.cn id=151 lang=python3
#
# [151] 颠倒字符串中的单词
#

# @lc code=start
class Solution:
    def reverseWords(self, s: str) -> str:
        res = []
        n = len(s)
        word = ""
        for i in range(n):
            if s[i] == " " and len(word) != 0:
                res.append(word)
                word = ""
            else:
                if s[i] == " ":
                    continue
                word += s[i]
        if len(word) != 0:
            res.append(word)
        return " ".join(res[::-1])
print(Solution().reverseWords("EPY2giL"))
# @lc code=end

