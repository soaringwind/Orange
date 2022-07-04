#
# @lc app=leetcode.cn id=53 lang=python3
#
# [53] 最大子数组和
#

# @lc code=start
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 动态规划n
        # if not nums:
        #     return
        # res = []
        # for i in range(len(nums)):
        #     if not res:
        #         res.append(nums[0])
        #         continue
        #     res.append(max(nums[i], res[i-1]+nums[i]))
        # return max(res)

        # 线段树
        # class wtevTree(object):
        #     def __init__(self, l, r, i, m) -> None:
        #         self.lSum = l
        #         self.rSum = r
        #         self.iSum = i
        #         self.mSum = m

        # def get_info(nums, left, right):
        #     if right - left < 1:
        #         return wtevTree(nums[left], nums[left], nums[left], nums[left])
        #     mid = (left+right) >> 1
        #     leftT = get_info(nums, left, mid)
        #     rightT = get_info(nums, mid+1, right)
        #     return push_up(leftT, rightT)

        # def push_up(leftT, rightT):
        #     l = max(leftT.lSum, leftT.iSum+rightT.lSum)
        #     r = max(leftT.rSum+rightT.iSum, rightT.rSum)
        #     i = leftT.iSum + rightT.iSum
        #     m = max(leftT.rSum + rightT.lSum, max(leftT.mSum, rightT.mSum))
        #     return wtevTree(l, r, i, m)

        # return get_info(nums, 0, len(nums)-1).mSum

        # DP
        # if not nums:
        #     return
        # dp = []
        # res = float("-inf")
        # for i in range(len(nums)):
        #     if i == 0:
        #         dp.append(nums[i])
        #     else:
        #         dp.append(nums[i] if dp[i-1]+nums[i] <= nums[i] else dp[i-1]+nums[i])
        #     res = max(res, dp[i])
        # return res

        # 前缀和数组
        pre_sum = [0]
        for i in range(len(nums)):
            pre_sum.append(nums[i]+pre_sum[i])
        min_pre_sum = float("inf")
        res = float("-inf")
        for i in range(len(nums)):
            min_pre_sum = min(min_pre_sum, pre_sum[i])
            res = max(res, pre_sum[i+1]-min_pre_sum)
        return res
# @lc code=end

