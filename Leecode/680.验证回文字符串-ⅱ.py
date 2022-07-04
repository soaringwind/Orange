#
# @lc app=leetcode.cn id=680 lang=python3
#
# [680] 验证回文字符串 Ⅱ
#

# @lc code=start
class Solution:
    def validPalindrome(self, s: str) -> bool:
        # if s == s[::-1]:
        #     return True
        # n = len(s)
        # for i in range(n):
        #     word = ""
        #     word += s[:i] + s[i+1:]
        #     if word == word[::-1]:
        #         return True
        # return False
        n = len(s)
        left = 0
        right = n - 1
        skip = 0
        while left < right:
            if s[left] != s[right]:
                if s[left] == s[right-1] and skip < 2:
                    skip += 1
                    new_left = left
                    new_right = right - 1
                    while new_left < new_right:
                        if s[new_left] == s[new_right]:
                            new_right -= 1
                            new_left += 1
                        else:
                            skip += 1
                            break
                    if skip < 2:
                        return True
                if s[left+1] == s[right]:
                    new_left = left + 1
                    new_right = right
                    while new_left < new_right:
                        if s[new_left] == s[new_right]:
                            new_right -= 1
                            new_left += 1
                        else:
                            return False
                    return True
                return False
            else:
                right -= 1
                left += 1
        return True
print(Solution().validPalindrome("aba"))
# @lc code=end

