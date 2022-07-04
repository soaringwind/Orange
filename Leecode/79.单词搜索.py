#
# @lc app=leetcode.cn id=79 lang=python3
#
# [79] 单词搜索
#

# @lc code=start
from typing import List


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])
        word_len = len(word)
        def dfs(tem_i, tem_j, tem_word):
            if tem_word >= word_len:
                return True
            if (tem_i, tem_j) in visited:
                return
            if tem_i >= m or tem_j >= n or tem_i < 0 or tem_j < 0:
                return False
            if board[tem_i][tem_j] == word[tem_word]:
                visited.add((tem_i, tem_j))
                if dfs(tem_i-1, tem_j, tem_word+1) or dfs(tem_i+1, tem_j, tem_word+1) or dfs(tem_i, tem_j-1, tem_word+1) or dfs(tem_i, tem_j+1, tem_word+1):
                    return True
                visited.remove((tem_i, tem_j))
            else:
                return False
        for i in range(m):
            for j in range(n):
                visited = set()
                if dfs(i, j, 0):
                    return True
        return False
# print(Solution().exist([["a","a","a"],["A","A","A"],["a","a","a"]], "aAaaaAaaA"))
# @lc code=end

