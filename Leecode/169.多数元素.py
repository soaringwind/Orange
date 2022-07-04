#
# @lc app=leetcode.cn id=169 lang=python3
#
# [169] 多数元素
#

# @lc code=start
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # res = nums[0]
        # count = 0
        # for i in nums:
        #     if i == res:
        #         count += 1
        #     elif count > 0:
        #         count -= 1
        #     else:
        #         res = i
        #         count = 1
        # return res
        nums.sort()
        return nums[len(nums)//2]
# @lc code=end

