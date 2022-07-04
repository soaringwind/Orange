#
# @lc app=leetcode.cn id=136 lang=python3
#
# [136] 只出现一次的数字
#

# @lc code=start
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # x = nums[0]
        # for i in nums[1:]:
        #     x = x ^ i
        # return x
        x = 0
        for i in nums:
            x = x ^ i
        return x
# @lc code=end

