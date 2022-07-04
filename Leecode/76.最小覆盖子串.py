#
# @lc app=leetcode.cn id=76 lang=python3
#
# [76] 最小覆盖子串
#

# @lc code=start


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # t_dic = {}
        # s_dic = {}
        # for i in range(len(t)):
        #     s_dic[t[i]] = 0
        #     if t[i] not in t_dic:
        #         t_dic[t[i]] = 1
        #     else:
        #         t_dic[t[i]] += 1
        # left = 0
        # right = 0
        # res = float("inf")
        # res_list = None
        # while right < len(s):
        #     if s[right] not in s_dic:
        #         s_dic[s[right]] = 1
        #     else:
        #         s_dic[s[right]] += 1
        #     flag = True
        #     for i in t:
        #         if t_dic[i] > s_dic[i]:
        #             flag = False
        #             break
        #     if not flag:
        #         right += 1
        #     else:
        #         res = min(res, right-left)
        #         if res == right-left:
        #             res_list = [left, right]
        #         s_dic[s[right]] -= 1
        #         s_dic[s[left]] -= 1
        #         left += 1
        # if res_list:
        #     return s[res_list[0]:res_list[1]+1]
        # else:
        #     return ""

        # 1. 滑动窗口
        # 构建最小窗口 -> 增加right -> 缩小left
        # [left, right)
        window = {}
        target = {}
        for i in s:
            if i not in window:
                window[i] = 0
            window[i] += 1
        for i in t:
            if i not in target:
                target[i] = 0
            target[i] += 1
        def is_valid(window):
            for i in target:
                if i not in window or window[i] < target[i]:
                    return False
            return True
        if not is_valid(window):
            return ""
        window = {}
        left = 0
        right = 0
        n = len(s)
        res = n
        res_right = 0
        res_left = 0
        while left < n:
            flag = False
            while not is_valid(window) and right < n:
                if s[right] not in window:
                    window[s[right]] = 0
                window[s[right]] += 1
                right += 1
            while is_valid(window):
                flag = True
                win_len = right - left
                if win_len <= res:
                    res = win_len
                    res_right = right
                    res_left = left
                window[s[left]] -= 1
                left += 1
            if not flag:
                left += 1
        return s[res_left:res_right]
s = "cabeca"
t = "cae"
print(Solution().minWindow(s, t))
# @lc code=end

