#
# @lc app=leetcode.cn id=33 lang=python3
#
# [33] 搜索旋转排序数组
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
        # right = len(nums) - 1
        # while left <= right:
        #     mid = (right+left) >> 1
        #     if nums[mid] == target:
        #         return mid
        #     if nums[left] <= nums[mid]:
        #         # left-mid为有序区间
        #         if target < nums[mid] and nums[left] <= target:
        #             right = mid - 1
        #         else:
        #             left = mid + 1
        #     else:
        #         # mid-right为有序区间
        #         if nums[mid] < target and target <= nums[right]:
        #             left = mid + 1
        #         else:
        #             right = mid - 1
        # return -1
        left = 0
        right = len(nums)
        while left < right:
            mid = (left + right) >> 1
            if nums[mid] == target:
                return mid
            if nums[left] < nums[mid]:
                if target < nums[mid] and target >= nums[left]:
                    right = mid
                else:
                    left = mid + 1
            else:
                if target > nums[mid] and target <= nums[right-1]:
                    left = mid + 1
                else:
                    right = mid
        return -1
            
# @lc code=end

