#
# @lc app=leetcode.cn id=45 lang=python3
#
# [45] 跳跃游戏 II
#

# @lc code=start
from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        # right = len(nums)-1
        # step = 0
        # while right > 0:
        #     for i in range(right):
        #         if (i+nums[i]) >= right:
        #             step += 1
        #             right = i
        #             break
        # return step
        # max_pos = 0
        # end = 0
        # step = 0
        # for i in range(len(nums)-1):
        #     if max_pos >= i:
        #         max_pos = max(max_pos, i+nums[i])
        #         if i == end:
        #             end = max_pos
        #             step += 1
        # return step
        step = 0
        end = 0
        max_pos = 0
        nums_len = len(nums)
        for i in range(nums_len):
            if end >= nums_len-1:
                return step
            max_pos = max(max_pos, i+nums[i])
            if i == end:
                step += 1
                end = max_pos
# @lc code=end

