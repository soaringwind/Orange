#
# @lc app=leetcode.cn id=349 lang=python3
#
# [349] 两个数组的交集
#

# @lc code=start
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        res = {}
        res_ = set()
        for i in nums1:
            if i not in res:
                res[i] = 0
            res[i] += 1
        for j in nums2:
            if j in res:
                res_.add(j)
        return list(res_)
# @lc code=end

