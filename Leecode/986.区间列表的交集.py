#
# @lc app=leetcode.cn id=986 lang=python3
#
# [986] 区间列表的交集
#

# @lc code=start
from typing import List


class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        # res = []
        # for start_1, end_1 in firstList:
        #     for start_2, end_2 in secondList:
        #         if end_1 < start_2:
        #             break
        #         if max(start_1, start_2) <= min(end_1, end_2):
        #             res.append([max(start_1, start_2), min(end_1, end_2)])
        # return res

        problem = firstList+secondList
        problem.sort(key=lambda x: (x[0], x[1]))
        res = []
        max_end1 = -1
        for i in range(len(problem)-1):
            start1, end1 = problem[i]
            max_end1 = max(max_end1, end1)
            start2, end2 = problem[i+1]
            if max_end1 < start2:
                continue
            if max(start1, start2) <= min(max_end1, end2):
                res.append([max(start1, start2), min(max_end1, end2)])
        return res
# print(Solution().intervalIntersection([[8,15]], [[2,6],[8,10],[12,20]]))
# @lc code=end

