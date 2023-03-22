# class Solution():
#     def __init__(self, nums) -> None:
#         # 暴力
#         res = 0
#         for i in range(len(nums)):
#             for j in range(i+1, len(nums)):
#                 res = max(res, min(nums[i], nums[j])*(j-i))
#         return res 

#         # 左边遍历，右边从头找，比它大的就停下，没有必要，

#         # 双指针
#         res = 0 
#         left = 0
#         right = len(nums)-1
#         while left < right:
#             res = max(res, min(nums[left, right])*(right-left))
#             if nums[left] < nums[right]:
#                 left += 1
#             else:
#                 right -= 1 
#         return res  

# 【斗地主之顺子】在斗地主扑克牌游戏中，扑克牌由小到大的顺序为：3,4,5,6,7,8,9,10,J,Q,K,A,2，玩家可以出的扑克牌阵型有：单张、对子、顺子、飞机、炸弹
# 等。其中顺子的出牌规则为：由至少5张由小到大连续递增的扑克牌组成，且不能包含2。
# 例如：{3,4,5,6,7}、{3,4,5,6,7,8,9,10,J,Q,K,A}都是有效的顺子；而{J,Q,K,A,2}、 {2,3,4,5,6}、{3,4,5,6}、{3,4,5,6,8}等都不是顺子。
# 给定一个包含13张牌的数组，如果有满足出牌规则的顺子，请输出顺子。
# 如果存在多个顺子，请每行输出一个顺子，且需要按顺子的第一张牌的大小（必须从小到大）依次输出。
# 如果没有满足出牌规则的顺子，请输出No。
# 输入描述：
# 	13张任意顺序的扑克牌，每张扑克牌数字用空格隔开，每张扑克牌的数字都是合法的，并且不包括大小王：
# 2 9 J 2 3 4 K A 7 9 A 5 6
# 不需要考虑输入为异常字符的情况。
# 输出描述：
# 	组成的顺子，每张扑克牌数字用空格隔开：3 4 5 6 7
# 示例：
# 输入：
# 	2 9 J 2 3 4 K A 7 9 A 5 6
# 输出：
# 3 4 5 6 7

class Solution(object):
    def __init__(self) -> None:
        nums = input().split()
        # 1. 替换jqk
        nums_new = []
        check = {
            11: "J", 
            12: "Q", 
            13: "K", 
            14: "A"
        }
        num_check = {}
        for i in range(len(nums)):
            if nums[i].isdigit() and nums[i] != "2":
                nums_new.append(int(nums[i]))
            elif nums[i] == "J":
                nums_new.append(11)
            elif nums[i] == "Q":
                nums_new.append(12)
            elif nums[i] == "K":
                nums_new.append(13)
            elif nums[i] == "A":
                nums_new.append(14)
            else:
                continue
        # 2. 排序(重复的不知道是否计算，如果计算则用字典储存数量即可)
        for i in range(len(nums_new)):
            if nums_new[i] not in num_check:
                num_check[nums_new[i]] = 0
            num_check[nums_new[i]] += 1
        nums_new = list(set(nums_new))
        nums_new.sort()
        # 3. 双指针输出
        for i in range(len(nums_new)):
            right = i+4
            long_tem = []
            while right < len(nums_new) and nums_new[i]+right-i == nums_new[right]:
                if nums_new[right] > 10:
                    tem = []
                    for j in range(i, right+1):
                        if nums_new[j] in check:
                            tem.append(check[nums_new[j]])
                        else:
                            tem.append(nums_new[j])
                    long_tem.append(tem)
                else:
                    long_tem.append(nums_new[i:right+1])
                right += 1
            if long_tem:
                print(("".join(map(str, long_tem[-1]))))


Solution()