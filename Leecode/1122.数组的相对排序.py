#
# @lc app=leetcode.cn id=1122 lang=python3
#
# [1122] 数组的相对排序
#

# @lc code=start
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        # 计数排序
        # rank = [0] * (max(arr1)+1)
        # res = []
        # for num in arr1:
        #     rank[num] += 1
        # for num in arr2:
        #     for i in range(rank[num]):
        #         res.append(num)
        #     rank[num] = 0
        # for i in range(len(rank)):
        #     if rank[i] != 0:
        #         for j in range(rank[i]):
        #             res.append(i)
        #         rank[i] = 0
        # return res

        # dict+sort
        arr2_dic = {}
        for i in arr2:
            if i not in arr2_dic:
                arr2_dic[i] = 0
        arr1.sort()
        tem = []
        for i in arr1:
            if i in arr2_dic:
                arr2_dic[i] += 1
            else:
                tem.append(i)
        res = []
        for i in arr2:
            while arr2_dic[i] > 0:
                res.append(i)
                arr2_dic[i] -= 1
        return res+tem
# @lc code=end

