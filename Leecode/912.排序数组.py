#
# @lc app=leetcode.cn id=912 lang=python3
#
# [912] 排序数组
#

# @lc code=start
from typing import List


class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # 多种排序算法：冒泡/归并
        nums.sort()
        return nums
        # 冒泡超出时间限制
        # def bubble(lo, hi):
        #     sort = True
        #     for i in range(lo+1, hi):
        #         if nums[i-1] > nums[i]:
        #             sort = False
        #             tem = nums[i-1]
        #             nums[i-1] = nums[i]
        #             nums[i] = tem
        #     return sort
        # lo = 0
        # hi = len(nums)
        # while lo < hi and not bubble(lo, hi):
        #     hi -= 1
        # return nums
        # 归并
        # def merge_sort(lo, hi):
        #     if hi - lo < 2:
        #         return 
        #     mid = (hi+lo) // 2
        #     merge_sort(lo, mid)
        #     merge_sort(mid, hi)
        #     merge(lo, mid, hi)
            
        # def merge(lo, mid, hi):
        #     tem = []
        #     left = lo
        #     right = mid
        #     while left < mid or right < hi:
        #         if not (left < mid) and not (right < hi):
        #             break
        #         elif left < mid and (hi <= right or nums[left] <= nums[right]):
        #             tem.append(nums[left])
        #             left += 1
        #         elif right < hi and (mid <= left or nums[right] < nums[left]):
        #             tem.append(nums[right])
        #             right += 1
        #     for i in range(len(tem)):
        #         nums[lo+i] = tem[i]

        # merge_sort(0, len(nums))
        # return nums

    # 个人思路遍历+二分

print(Solution().sortArray([5,1,1,2,0,0]))
# @lc code=end

