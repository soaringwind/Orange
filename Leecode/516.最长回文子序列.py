#
# @lc app=leetcode.cn id=516 lang=python3
#
# [516] 最长回文子序列
#

# @lc code=start
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # 暴力求解法
        # n = len(s)
        # res = 0
        # for i in range(2**n):
        #     bin_str = bin(i)[2:][::-1]
        #     tem = ""
        #     for j in range(len(bin_str)):
        #         if bin_str[j] == "1":
        #             tem += s[j]
        #     if tem == tem[::-1]:
        #         res = max(res, len(tem))
        # return res

        # 动态规划
        # n =  len(s)
        # res = [[0]*n for _ in range(n)]
        # for i in range(n-1, -1, -1):
        #     res[i][i] = 1
        #     for j in range(i+1, n):
        #         if s[i] == s[j]:
        #             res[i][j] = res[i+1][j-1] + 2
        #         else:
        #             res[i][j] = max(res[i+1][j], res[i][j-1])
        # return res[0][n-1]

        # 递归
        # n = len(s)
        # if n == 1:
        #     return 1
        # res = [[-1]*n for _ in range(n)]
        # def dp(s, i, j):
        #     if j < i:
        #         return 0
        #     if i == j:
        #         return 1
        #     if res[i][j] != -1:
        #         return res[i][j]
        #     if s[i] == s[j]:
        #         return 2+dp(s, i+1, j-1)
        #     if s[i] != s[j]:
        #         return max(dp(s, i+1, j), dp(s, i, j-1))
        # for k in range(n-1, -1, -1):
        #     for v in range(k+1, n):
        #         res[k][v] = dp(s, k, v)
        # return res[0][n-1]

        # 最长公共子串
        text1 = s
        text2 = s[::-1]
        dp = [[0]*(len(s)+1) for _ in range(len(s)+1)]
        for i in range(len(s)):
            for j in range(len(s)):
                if text1[i] == text2[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        # print(dp)
        return dp[i+1][j+1]
# print(Solution().longestPalindromeSubseq("bbbab"))
# @lc code=end

