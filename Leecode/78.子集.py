#
# @lc app=leetcode.cn id=78 lang=python3
#
# [78] 子集
#
# 用二进制表示是否在子集内
# @lc code=start
from typing import List


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # res = []
        # for mask in range(2**len(nums)):
        #     tem = []
        #     for i in range(len(nums)):
        #         if mask & 2**i:
        #             tem.append(nums[i])
        #     res.append(tem)
        # return res

        tem = []
        res = []
        def dfs(idx):
            res.append(tem.copy())
            for i in range(idx, len(nums)):
                tem.append(nums[i])
                dfs(i+1)
                tem.pop()
            return
        dfs(0)
        return res
# print(Solution().subsets([1,2,3]))
# dfs的解法需要思考



# @lc code=end

