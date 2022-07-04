#
# @lc app=leetcode.cn id=417 lang=python3
#
# [417] 太平洋大西洋水流问题
#

# @lc code=start
from typing import List


class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        # dfs
        # atl_visited = set()
        # pac_visited = set()
        # res = []
        # tem = []
        # def run_to_atl(x, y):
        #     if x+1 >= len(heights) or y+1 >= len(heights[0]) or [x, y] in res:
        #         tem.append((x, y))
        #         return True
        #     if (x, y) in atl_visited or (x, y) in tem:
        #         return 
        #     tem.append((x, y))
        #     if y+1 < len(heights[0]) and heights[x][y] >= heights[x][y+1]:
        #         if run_to_atl(x, y+1):
        #             return True
        #     if x-1 >= 0 and heights[x][y] >= heights[x-1][y]:
        #         if run_to_atl(x-1, y):
        #             return True
        #     if y-1 >= 0 and heights[x][y] >= heights[x][y-1]:
        #         if run_to_atl(x, y-1):
        #             return True
        #     if x+1 < len(heights) and heights[x][y] >= heights[x+1][y]:
        #         if run_to_atl(x+1, y):
        #             return True
        #     return
        # def run_to_pac(x, y):
        #     if x-1 < 0 or y-1 < 0 or [x, y] in res:
        #         tem.append((x, y))
        #         return True
        #     if (x, y) in pac_visited or (x, y) in tem:
        #         return 
        #     tem.append((x, y))
        #     if y+1 < len(heights[0]) and heights[x][y] >= heights[x][y+1]:
        #         if run_to_pac(x, y+1):
        #             return True
        #     if x-1 >= 0 and heights[x][y] >= heights[x-1][y]:
        #         if run_to_pac(x-1, y):
        #             return True
        #     if y-1 >= 0 and heights[x][y] >= heights[x][y-1]:
        #         if run_to_pac(x, y-1):
        #             return True
        #     if x+1 < len(heights) and heights[x][y] >= heights[x+1][y]:
        #         if run_to_pac(x+1, y):
        #             return True
        #     return
        # a = len(heights)
        # b = len(heights[0])
        # for i in range(a):
        #     for j in range(b):
        #         tem = []
        #         flag_1 = run_to_atl(i, j)
        #         tem = []
        #         flag_2 = run_to_pac(i, j)
        #         if flag_1 and flag_2:
        #             res.append([i, j])
        # return res
        m, n = len(heights), len(heights[0])
        def search(starts):
            visited = set()
            def dfs(x, y):
                if (x, y) in visited:
                    return
                visited.add((x, y))
                for nx, ny in ((x, y+1), (x, y-1), (x-1, y), (x+1, y)):
                    if 0 <= nx < m and 0 <= ny < n and heights[nx][ny] >= heights[x][y]:
                        dfs(nx, ny)
            for x, y in starts:
                dfs(x, y)
            return visited
        pacific = [(0, i) for i in range(n)] + [(i, 0) for i in range(1, m)]
        atlantic = [(m-1, i) for i in range(n)] + [(i, n-1) for i in range(m-1)]
        return list(map(list, search(pacific) & search(atlantic)))
heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
print(Solution().pacificAtlantic(heights=heights))
# @lc code=end

