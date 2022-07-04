#
# @lc app=leetcode.cn id=39 lang=python3
#
# [39] 组合总和
#

# @lc code=start
from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        tem = []
        res_list = []
        def dfs(i):
            if sum(tem) == target:
                # tem_ = tem.copy()
                # tem_.sort()
                # res_list[str(tem_)] = tem_
                res_list.append(tem.copy())
                return
            if sum(tem) > target:
                return
            for j in range(i, len(candidates)):
                if sum(tem) > target:
                    break
                tem.append(candidates[j])
                dfs(j)
                tem.pop()
            return
        dfs(0)
        return res_list

# print(Solution().combinationSum([1, 2], 4))
# @lc code=end

