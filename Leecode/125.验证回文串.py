#
# @lc app=leetcode.cn id=125 lang=python3
#
# [125] 验证回文串
#

# @lc code=start
class Solution:
    def isPalindrome(self, s: str) -> bool:
        left = 0
        n = len(s)        
        right = n - 1
        while left < right:
            if s[left].isalpha() or s[left].isdigit():
                left_al = s[left].capitalize()
                if s[right].isalpha() or s[right].isdigit():
                    right_al = s[right].capitalize()
                    if left_al != right_al:
                        return False
                    else:
                        left += 1
                        right -= 1
                else:
                    right -= 1
            else:
                left += 1
        return True
# print(Solution().isPalindrome("A man, a plan, a canal: Panama"))
# @lc code=end

