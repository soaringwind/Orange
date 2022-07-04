#
# @lc app=leetcode.cn id=528 lang=python3
#
# [528] 按权重随机选择
#

# @lc code=start
class Solution:

    def __init__(self, w: List[int]):
        self.w = [sum(w[:i+1]) for i in range(len(w))]
        self.s = sum(w)


    def pickIndex(self) -> int:
        import random
        choice = random.randint(0, self.s-1)
        for i in range(len(self.w)):
            if choice < self.w[i]:
                return i



# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()
# @lc code=end

