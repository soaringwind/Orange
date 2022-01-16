#
# @lc app=leetcode.cn id=875 lang=python3
#
# [875] 爱吃香蕉的珂珂
#

# @lc code=start
from typing import List


class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        piles_sum = sum(piles)
        max_k = max(piles)
        min_k = piles_sum // h if piles_sum%h==0 else (piles_sum // h)+1
        while min_k < max_k:
            mid_k = (min_k+max_k) // 2
            sum_h = 0
            for pile in piles:
                if pile%mid_k == 0:
                    sum_h += pile // mid_k
                else:
                    sum_h += (pile // mid_k) + 1
            if sum_h <= h:
                max_k = mid_k
            else:
                min_k = mid_k + 1
        return min_k
piles = [30,11,23,4,20]
h = 5
print(Solution().minEatingSpeed(piles, h))
# @lc code=end

