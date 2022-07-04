#
# @lc app=leetcode.cn id=394 lang=python3
#
# [394] 字符串解码
#

# @lc code=start
class Solution:
    def decodeString(self, s: str) -> str:
        # 1. 递归解决
        # n = len(s)
        # res = ""
        # res_list = []
        # global left
        # left = 0
        # def decodemd():
        #     new_res = ""
        #     while True:
        #         global left
        #         res_list.append(left)
        #         if s[left] == ']':
        #             break
        #         if s[left] == '[':
        #             left += 1
        #             continue
        #         if s[left].isalpha():
        #             new_res += s[left]
        #         if s[left].isdigit():
        #             num = 0
        #             while s[left].isdigit():
        #                 num = 10*num + int(s[left])
        #                 left += 1
        #             new_res += num * decodemd()
        #             left = max(res_list)
        #         left += 1
        #     return new_res
        # while left < n:
        #     if s[left].isdigit():
        #         num = 0
        #         while s[left].isdigit():
        #             num = 10*num + int(s[left])
        #             left += 1
        #         res += num * decodemd()
        #         left = max(res_list)
        #     if s[left].isalpha():
        #         res += s[left]
        #     left += 1
        # return res
        # 2. 辅助栈
        n = len(s)
        res = []
        left = 0
        while left < n:
            if s[left].isalpha():
                res.append(s[left])
            elif s[left].isalnum():
                num = 0
                while s[left].isalnum():
                    num = 10*num + int(s[left])
                    left += 1
                res.append(num)
                continue
            elif s[left] == '[':
                res.append(s[left])
            else:
                support_res = ""
                res_str = res.pop()
                while res_str != '[':
                    support_res += res_str
                    res_str = res.pop()
                for _ in range(res.pop()):
                    for word in support_res[::-1]:
                        res.append(word)
            left += 1
        return ''.join(res)
# print(Solution().decodeString("3[z]2[2[y]pq4[2[jk]e1[f]]]ef"))
# @lc code=end

