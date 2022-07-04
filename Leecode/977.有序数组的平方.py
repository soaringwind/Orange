#
# @lc app=leetcode.cn id=977 lang=python3
#
# [977] 有序数组的平方
#

# @lc code=start
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        # left = 0
        # right = len(nums) -1
        # tem = []
        # while left <= right:
        #     if nums[left] ** 2 <= nums[right] ** 2:
        #         tem.insert(0, nums[right]**2)
        #         right -= 1
        #     else:
        #         tem.insert(0, nums[left]**2)
        #         left += 1
        # return tem
        left = 0
        right = len(nums) - 1
        tem = []
        while left <= right:
            if nums[left]**2 <= nums[right]**2:
                tem.append(nums[right]**2)
                right -= 1
            else:
                tem.append(nums[left]**2)
                left += 1
        return tem[::-1]
# @lc code=end

