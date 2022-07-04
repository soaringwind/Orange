#
# @lc app=leetcode.cn id=56 lang=python3
#
# [56] 合并区间
#

# @lc code=start
from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # end >= start
        res = []
        intervals.sort(key=lambda x: x[0])
        pre_end = -1
        for start, end in intervals:
            if pre_end != -1 and pre_end >= start:
                tem = res.pop()
                res.append([tem[0], end if tem[1] <= end else tem[1]])
            else:
                res.append([start, end])
            pre_end = res[-1][1]
        return res
# print(Solution().merge([[0,0],[1,2],[5,5],[2,4],[3,3],[5,6],[5,6],[4,6],[0,0],[1,2],[0,2],[4,5]]))
# @lc code=end

