#
# @lc app=leetcode.cn id=155 lang=python3
#
# [155] 最小栈
#

# @lc code=start
class MinStack:

    def __init__(self):
        self.res = []
        # 辅助栈
        self.min_res = []


    def push(self, val: int) -> None:
        if not self.min_res:
            self.min_res.append(val)
        else:
            self.min_res.append(val if val < self.min_res[-1] else self.min_res[-1])
        self.res.append(val)

    def pop(self) -> None:
        self.min_res.pop()
        return self.res.pop()

    def top(self) -> int:
        return self.res[-1]

    def getMin(self) -> int:
        return self.min_res[-1]



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
# @lc code=end

