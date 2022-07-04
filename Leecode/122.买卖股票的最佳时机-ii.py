#
# @lc app=leetcode.cn id=122 lang=python3
#
# [122] 买卖股票的最佳时机 II
#

# @lc code=start
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_pro = 0
        min_price = float("inf")
        for price in prices:
            min_price = min(min_price, price)
            if min_price <= price:
                max_pro += price - min_price
                min_price = price
        return max_pro
# @lc code=end

