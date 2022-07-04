#
# @lc app=leetcode.cn id=464 lang=python3
#
# [464] 我能赢吗
#

# @lc code=start
class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        # dfs深度搜索
        # used_num_dic = {}
        # used_num = 0
        # if (1+maxChoosableInteger) * maxChoosableInteger // 2 < desiredTotal:
        #     return False
        # def dfs(used_num, cur_val):
        #     if used_num in used_num_dic:
        #         print(used_num_dic)
        #         return used_num_dic[used_num]
        #     for i in range(maxChoosableInteger):
        #         if (used_num >> i) & 1 != 0:
        #             continue
        #         if cur_val+i+1 >= desiredTotal or not dfs((used_num | 1 << i), cur_val+i+1):
        #             # used_num[i] = "0"
        #             used_num_dic[used_num] = True
        #             return True
        #         # if not dfs(used_num, cur_val+i+1):
        #             # used_num_dic[used_num] = True
        #             # used_num[i] = "0"
        #             # return True
        #         # used_num_dic[used_num] = False
        #         # used_num[i] = "0"
        #     # used_num_dic["".join(used_num)] = False
        #     return False
        # return dfs(used_num, 0)

        used_num_dic = {}
        used_num = ["0"]*maxChoosableInteger
        if (1+maxChoosableInteger) * maxChoosableInteger // 2 < desiredTotal:
            return False
        def dfs(used_num, cur_val):
            # print(used_num)
            if "".join(used_num) in used_num_dic:
            #     print(used_num_dic)
                return used_num_dic["".join(used_num)] != 0
            for i in range(maxChoosableInteger):
                if used_num[i] == "1":
                    continue
                used_num[i] = "1"
                if cur_val+i+1 >= desiredTotal or not dfs(used_num, cur_val+i+1):
                    used_num[i] = "0"
                    used_num_dic["".join(used_num)] = True
                    return True
                # if not dfs(used_num, cur_val+i+1):
                    # used_num_dic[used_num] = True
                used_num[i] = "0"
                    # return True
                # used_num_dic[used_num] = False
                # used_num[i] = "0"
            # used_num_dic["".join(used_num)] = False
            return False
        return dfs(used_num, 0)
# print(Solution().canIWin(10, 11))
# @lc code=end

