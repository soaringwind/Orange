#
# @lc app=leetcode.cn id=16 lang=python3
#
# [16] 最接近的三数之和
#

# @lc code=start
from typing import List


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        n = len(nums)
        res = float("inf")
        nums.sort()
        for i in range(n):
            left = i + 1
            right = n - 1
            while left < right:
                nums_sum = nums[i] + nums[left] + nums[right]
                if abs(nums_sum-target) < abs(res - target):
                    res = nums_sum
                if nums_sum < target:
                    left += 1
                elif nums_sum > target:
                    right -= 1
                else:
                    return res
        return res
nums = [1,1,1,0]
target = -100
print(Solution().threeSumClosest(nums, target))
# @lc code=end

