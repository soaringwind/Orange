#
# @lc app=leetcode.cn id=123 lang=python3
#
# [123] 买卖股票的最佳时机 III
#

# @lc code=start
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[[0]*2 for _ in range(3)] for _ in range(len(prices))]
        for i in range(len(prices)):
            for j in range(1, 3):
                if i == 0:
                    dp[i][j][0] = 0
                    dp[i][j][1] = -prices[i]
                    continue
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
                dp[i][j][1] = max(dp[i-1][j-1][0] - prices[i], dp[i-1][j][1])
        return dp[i][j][0]
# print(Solution().maxProfit([3,3,5,0,0,3,1,4]))
# @lc code=end

