#
# @lc app=leetcode.cn id=121 lang=python3
#
# [121] 买卖股票的最佳时机
#

# @lc code=start
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_val = float("inf")
        res = float("-inf")
        for i in range(len(prices)):
            if  prices[i] < min_val:
                min_val = prices[i]
            res = max(prices[i]-min_val, res)
        return res
# @lc code=end

