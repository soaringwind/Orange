#
# @lc app=leetcode.cn id=1 lang=python3
#
# [1] 两数之和
#

# @lc code=start
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_dic = {}
        n = len(nums)
        for i in range(n):
            if target - nums[i] in nums_dic:
                return [i, nums_dic[target-nums[i]]]
            nums_dic[nums[i]] = i
# @lc code=end

