#
# @lc app=leetcode.cn id=704 lang=python3
#
# [704] 二分查找
#

# @lc code=start


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 线性查找/二分查找
        # for i in range(len(nums)):
        #     if nums[i] == target:
        #         return i
        # return -1
        # left = 0
        # right = len(nums)
        # while 1 < right - left:
        #     mid = (left+right) >> 1
        #     if target < nums[mid]:
        #         right = mid
        #     else:
        #         left = mid
        # return left if nums[left] == target else -1

        # [left, right)
        # 表示当0 < right - left必须成立才可能有元素
        left = 0
        right = len(nums)
        while left < right:
            mid = (left+right) >> 1
            if target < nums[mid]:
                right = mid
            elif target > nums[mid]:
                left = mid + 1
            else:
                return mid
        return -1
# @lc code=end

