#
# @lc app=leetcode.cn id=51 lang=python3
#
# [51] N 皇后
#

# @lc code=start
from typing import List


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # res = set()
        # res_list = set()
        # visited_ = {}
        # head_visited = set()
        # for i in range(n):
        #     for j in range(n):
        #         visited_[(i, j)] = 0
        # def dfs(tem_i, tem_j, queen_list_len):
        #     # if visited[(tem_i, tem_j)] > 0:
        #     #     return False
        #     for v in range(n):
        #         # 横线
        #         # visited[(tem_i, v)] += 1
        #         # 竖线
        #         visited[(v, tem_j)] += 1
        #         # 斜线
        #         # if (tem_i, v) in queen_list or (v, tem_j) in queen_list or (tem_i+v, tem_j-v) in queen_list or (tem_i-v, tem_j+v) in queen_list or (tem_i-v, tem_j-v) in queen_list or (tem_i+v, tem_j+v) in queen_list:
        #         #     return False
        #         if 0 <= tem_i+v < n and 0 <= tem_j-v < n:
        #             visited[(tem_i+v, tem_j-v)] += 1
        #         # if 0 <= tem_i-v < n and 0 <= tem_j+v < n:
        #         #     visited[(tem_i-v, tem_j+v)] += 1
        #         # if 0 <= tem_i-v < n and 0 <= tem_j-v < n:
        #         #     visited[(tem_i-v, tem_j-v)] += 1
        #         if 0 <= tem_i+v < n and 0 <= tem_j+v < n:
        #             visited[(tem_i+v, tem_j+v)] += 1
        #     # for tem_tem_i, tem_tem_j in queen_list:
        #     #     if visited[(tem_tem_i, tem_tem_j)] > 6 or visited[(tem_i, tem_j)] > 6:
        #     #         return False
        #     if max(visited.values()) > 6:
        #         return False
        #     queen_list.append((tem_i, tem_j))
        #     # queen_list_ = sorted(queen_list, key=lambda x: x[0])
        #     if len(queen_list) == n:
        #         # queen_list_ = sorted(queen_list, key=lambda x: x[0])
        #         res_list.add(tuple(queen_list))
        #         return True
        #     for tem_i_ in range(tem_i+1, n):
        #         for tem_j_ in range(n):
        #             if visited[(tem_i_, tem_j_)] != 0 or (tem_i_, tem_j_) in head_visited:
        #                 continue
        #             dfs(tem_i_, tem_j_, queen_list_len+1)
        #                 # print(queen_list)
        #                 # num = 0
        #             if queen_list_len <= len(queen_list):
        #                 queen_list.pop()
        #                 # print(queen_list)
        #             for v in range(n):
        #                 # 横线
        #                 # visited[(tem_i_, v)] -= 1
        #                 # 竖线
        #                 visited[(v, tem_j_)] -= 1
        #                 # 斜线
        #                 if 0 <= tem_i_+v < n and 0 <= tem_j_-v < n:
        #                     visited[(tem_i_+v, tem_j_-v)] -= 1
        #                 # if 0 <= tem_i_-v < n and 0 <= tem_j_+v < n:
        #                 #     visited[(tem_i_-v, tem_j_+v)] -= 1
        #                 # if 0 <= tem_i_-v < n and 0 <= tem_j_-v < n:
        #                 #     visited[(tem_i_-v, tem_j_-v)] -= 1
        #                 if 0 <= tem_i_+v < n and 0 <= tem_j_+v < n:
        #                     visited[(tem_i_+v, tem_j_+v)] -= 1
        #     return
        # for i in range(1):
        #     for j in range(n):
        #         head_visited.add((i, j))
        #         queen_list = []
        #         visited = visited_.copy()
        #         dfs(i, j, 0)
        #         # for v in range(n):
        #         #     # 横线
        #         #     visited[(i, v)] -= 1
        #         #     # 竖线
        #         #     visited[(v, j)] -= 1
        #         #     # 斜线
        #         #     if 0 <= i+v < n and 0 <= j-v < n:
        #         #         visited[(i+v, j-v)] -= 1
        #         #     if 0 <= i-v < n and 0 <= j+v < n:
        #         #         visited[(i-v, j+v)] -= 1
        #         #     if 0 <= i-v < n and 0 <= j-v < n:
        #         #         visited[(i-v, j-v)] -= 1
        #         #     if 0 <= i+v < n and 0 <= j+v < n:
        #         #         visited[(i+v, j+v)] -= 1
        # # print(len(res_list))
        # for item in res_list:
        #     chase_board = [["." for i in range(n)] for j in range(n)]
        #     for tem_i, tem_j in item:
        #         chase_board[tem_i][tem_j] = "Q"
        #     res.add(tuple(["".join(chase_board[tem_i]) for tem_i in range(n)]))
        # return [list(item) for item in res]

        # 回溯算法
        res = []
        C = [-1]*n
        def search(cur):
            if cur == n:
                tem = []
                for i in range(n):
                    if C[i] != -1:
                        tem.append((i, C[i]))
                res.append(tem)
                return True
            else:
                for i in range(n):
                    ok = 1
                    C[cur] = i
                    for j in range(cur):
                        if C[cur] == C[j] or cur-C[cur] == j-C[j] or cur+C[cur] == j+C[j]:
                            ok = 0
                            C[cur] = -1
                            break
                    if ok == 1:
                        search(cur+1)
                        C[cur] = -1
            return
        search(0)
        res_list = []
        for item in res:
            chase_board = [["." for i in range(n)] for j in range(n)]
            for tem_i, tem_j in item:
                chase_board[tem_i][tem_j] = "Q"
            res_list.append(tuple(["".join(chase_board[tem_i]) for tem_i in range(n)]))
        return [list(item) for item in res_list]
        # return res

# print(Solution().solveNQueens(4))
# @lc code=end

