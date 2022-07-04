#
# @lc app=leetcode.cn id=454 lang=python3
#
# [454] 四数相加 II
#

# @lc code=start
from typing import List


class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        dic_1_2 = {}
        res = 0
        for i in nums1:
            for j in nums2:
                if i+j not in dic_1_2:
                    dic_1_2[i+j] = 0
                dic_1_2[i+j] += 1
        for i in nums3:
            for j in nums4:
                if -(i+j) in dic_1_2:
                    res += dic_1_2[-(i+j)]
        return res
# nums1 = [1,2]
# nums2 = [-2,-1]
# nums3 = [-1,2]
# nums4 = [0,2]
# print(Solution().fourSumCount(nums1, nums2, nums3, nums4))
# @lc code=end

