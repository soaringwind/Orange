#
# @lc app=leetcode.cn id=134 lang=python3
#
# [134] 加油站
#

# @lc code=start
from typing import List

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        start_stat = 0
        while start_stat < n:
            remain_gas = 0
            stat_num = 0
            now_stat = start_stat
            while stat_num < n:
                remain_gas += gas[now_stat]
                remain_gas -= cost[now_stat]
                now_stat = now_stat+1 if now_stat+1<n else n-now_stat-1
                stat_num += 1
                if remain_gas < 0:
                    if now_stat <= start_stat:
                        return -1
                    start_stat = now_stat
                    break
            if stat_num == n:
                return start_stat
        return -1

gas = [1,2,3,4,5]
cost = [3,4,5,1,2]
print(Solution().canCompleteCircuit(gas, cost))

# @lc code=end

