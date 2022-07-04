#
# @lc app=leetcode.cn id=219 lang=python3
#
# [219] 存在重复元素 II
#

# @lc code=start
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        res = {}
        for i in range(len(nums)):
            if nums[i] in res and abs(res[nums[i]] - i) <= k:
                return True
            res[nums[i]] = i
        return False
# @lc code=end

