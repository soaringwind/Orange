#
# @lc app=leetcode.cn id=242 lang=python3
#
# [242] 有效的字母异位词
#

# @lc code=start
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # 1. hash map
        # res = {}
        # for i in s:
        #     if i not in res:
        #         res[i] = 0
        #     res[i] += 1
        # for i in t:
        #     if i in res:
        #         res[i] -= 1
        #     else:
        #         return False
        # for j in res:
        #     if res[j] != 0:
        #         return False
        # return True

        # 2. list
        res = [0]*26
        for i in s:
            res[ord(i)-ord("a")] += 1
        for j in t:
            res[ord(j) - ord("a")] -= 1
        for num in res:
            if num != 0:
                return False
        return True
            
# @lc code=end

