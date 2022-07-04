#
# @lc app=leetcode.cn id=26 lang=python3
#
# [26] 删除有序数组中的重复项
#

# @lc code=start
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        left = 0
        n = len(nums)
        right = left
        while right < n:
            if nums[left] != nums[right]:
                left += 1
                nums[left] = nums[right]
            else:
                right += 1
        while left < n-1:
            nums.pop()
            left += 1
        # return left+1, nums
# @lc code=end

