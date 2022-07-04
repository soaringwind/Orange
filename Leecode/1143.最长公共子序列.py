#
# @lc app=leetcode.cn id=1143 lang=python3
#
# [1143] 最长公共子序列
#

# @lc code=start
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # 动态规划m*n
        # import numpy as np
        # dp = np.zeros(shape=(len(text1)+1, len(text2)+1))
        # m = len(text1)
        # n = len(text2)
        # dp = [[0] * (n+1) for _ in range(m+1)]
        # for i in range(1, len(text1)+1):
        #     for j in range(1, len(text2)+1):
        #         if text1[i-1] == text2[j-1]:
        #             dp[i][j] = dp[i-1][j-1] + 1
        #         else:
        #             dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        # return dp[i][j]

        # 动态规划
        m = len(text1)
        n = len(text2)
        dp = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    dp[i][j] += 1 if text1[i] == text2[j] else 0
                    continue
                if i == 0:
                    if dp[i][j-1] != 0 or text1[i] != text2[j]:
                        dp[i][j] = dp[i][j-1]
                    else:
                        dp[i][j] = 1
                    continue
                if j == 0:
                    if dp[i-1][j] != 0 or text1[i] != text2[j]:
                        dp[i][j] = dp[i-1][j]
                    else:
                        dp[i][j] = 1
                    continue
                dp[i][j] = dp[i-1][j-1] + 1 if text1[i] == text2[j] else max(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])
        return dp[i][j]
# @lc code=end

