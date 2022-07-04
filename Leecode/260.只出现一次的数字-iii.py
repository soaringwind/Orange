#
# @lc app=leetcode.cn id=260 lang=python3
#
# [260] 只出现一次的数字 III
#
from typing import List
# @lc code=start
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        # res = 0
        # res_list = []
        # nums.sort()
        # for i in range(len(nums)):
        #     res = res ^ nums[i]
        #     if i % 2 == 1 and res != 0:
        #         a = 0
        #         b = 0
        #         for j in range(i):
        #             a = a ^ nums[j]
        #         for k in range(i, len(nums)):
        #             b = b ^ nums[k]
        #         res_list.append(a)
        #         res_list.append(b)
        #         return res_list
        # dic = {}
        # for i in nums:
        #     if i not in dic:
        #         dic[i] = 0
        #     dic[i] += 1
        # return [x for x in dic if dic[x]==1]
        # xorsum = 0
        # for num in nums:
        #     xorsum ^= num
        # lsb = xorsum & -xorsum
        # type_1 = 0
        # type_2 = 0
        # for num in nums:
        #     if num&lsb:
        #         type_1 ^= num
        #     else:
        #         type_2 ^= num
        # return [type_1, type_2]
        nums_dic = {}
        for num in nums:
            if num not in nums_dic:
                nums_dic[num] = 0
            nums_dic[num] += 1
        res = []
        for num in nums_dic:
            if nums_dic[num] == 1:
                res.append(num)
        return res

# print(Solution().singleNumber([1,2,1,3,2,5]))
# @lc code=end

