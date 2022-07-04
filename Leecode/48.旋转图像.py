#
# @lc app=leetcode.cn id=48 lang=python3
#
# [48] 旋转图像
#

# @lc code=start
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # horizon and diagon
        n = len(matrix)
        for i in range(n//2):
            for j in range(n):
                tem = matrix[i][j]
                matrix[i][j] = matrix[n-1-i][j]
                matrix[n-1-i][j] = tem
        for i in range(n):
            for j in range(i+1, n):
                tem = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = tem
        return matrix

# @lc code=end

