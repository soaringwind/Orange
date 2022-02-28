#
# @lc app=leetcode.cn id=76 lang=python3
#
# [76] 最小覆盖子串
#

# @lc code=start
# from curses.ascii import SO
from operator import le


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        t_dic = {}
        s_dic = {}
        for i in range(len(t)):
            s_dic[t[i]] = 0
            if t[i] not in t_dic:
                t_dic[t[i]] = 1
            else:
                t_dic[t[i]] += 1
        left = 0
        right = 0
        res = float("inf")
        res_list = None
        while right < len(s):
            if s[right] not in s_dic:
                s_dic[s[right]] = 1
            else:
                s_dic[s[right]] += 1
            flag = True
            for i in t:
                if t_dic[i] > s_dic[i]:
                    flag = False
                    break
            if not flag:
                right += 1
            else:
                res = min(res, right-left)
                if res == right-left:
                    res_list = [left, right]
                s_dic[s[right]] -= 1
                s_dic[s[left]] -= 1
                left += 1
        if res_list:
            return s[res_list[0]:res_list[1]+1]
        else:
            return ""
s = "ab"
t = "a"
print(Solution().minWindow(s, t))
# @lc code=end

