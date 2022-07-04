#
# @lc app=leetcode.cn id=88 lang=python3
#
# [88] 合并两个有序数组
#

# @lc code=start
from typing import List


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # left = 0
        # right = 0
        # while left < m+n and right < n:
        #     if nums1[left] > nums2[right]:
        #         nums1.insert(left, nums2[right])
        #         nums1.pop()
        #         right += 1
        #     left += 1
        # for i in range(right, n):
        #     nums1[-(n-i)] = nums2[i]
        # return nums1

        # 反向思维：倒着插入
        left = m-1
        right = n-1
        cur = m+n-1
        while cur >= 0:
            if right < 0 or (left >= 0 and nums1[left] >= nums2[right]):
                nums1[cur] = nums1[left]
                left -= 1
            else:
                nums1[cur] = nums2[right]
                right -= 1
            cur -= 1
        return nums1
print(Solution().merge([2,0], 1, [1], 1))
# @lc code=end

