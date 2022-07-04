#
# @lc app=leetcode.cn id=137 lang=python3
#
# [137] 只出现一次的数字 II
#

# @lc code=start


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        nums_res = {}
        for i in nums:
            if i not in nums_res:
                nums_res[i] = 0
            nums_res[i] += 1
        for item in nums_res:
            if nums_res[item] == 1:
                return item
# @lc code=end

