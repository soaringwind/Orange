#
# @lc app=leetcode.cn id=475 lang=python3
#
# [475] 供暖器
#

# @lc code=start
from typing import List

class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        # houses.sort()
        # heaters.sort()
        # houses_dic = {}
        # res = float("-inf")
        # for i in range(len(houses)):
        #     houses_dic[i] = float("inf")
        #     for j in range(len(heaters)):
        #         houses_dic[i] = min(houses_dic[i], abs(heaters[j]-houses[i]))
        # for i in range(len(houses)):
        #     res = max(res, houses_dic[i])
        # return res
        heaters.sort()
        ans = 0
        for i in range(len(houses)):
            house = houses[i]
            j = bs(heaters, house)
            left_dis = abs(heaters[j] - house)
            right_dis = house - heaters[j-1] if j-1>=0 else float("inf")
            cur_dis = min(left_dis, right_dis)
            ans = max(ans, cur_dis)
        return ans
def bs(h_list, hou):
    left = 0
    right = len(h_list)-1
    while left < right:
        mid = (left+right)//2
        if h_list[mid] >= hou:
            right = mid
        if h_list[mid] < hou:
            left = mid + 1
    return right

houses = [1,5]
heaters = [2]
print(Solution().findRadius(houses, heaters))


# @lc code=end

