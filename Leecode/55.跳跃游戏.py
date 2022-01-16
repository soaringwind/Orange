#
# @lc app=leetcode.cn id=55 lang=python3
#
# [55] 跳跃游戏
#

# @lc code=start
from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_dis = 0
        for i in range(len(nums)):
            max_dis = max(max_dis, i+nums[i])
            if max_dis >= len(nums)-1:
                return True
            if nums[i] == 0 and max_dis == i:
                return False
# @lc code=end

