#
# @lc app=leetcode.cn id=1288 lang=python3
#
# [1288] 删除被覆盖区间
#

# @lc code=start
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: (x[0], -x[1]))
        res = 0
        min_left = float("inf")
        max_right = -1
        for start, end in intervals:
            if max_right == -1:
                min_left = min(min_left, start)
                max_right = max(max_right, end)
                res += 1
                continue
            if min_left <= start and end <= max_right:
                continue
            else:
                min_left = min(min_left, start)
                max_right = max(max_right, end)
                res += 1
        return res

# @lc code=end

