#
# @lc app=leetcode.cn id=173 lang=python3
#
# [173] 二叉搜索树迭代器
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.res = []
        def tranvse(root):
            if not root:
                return
            tranvse(root.left)
            self.res.append(root.val)
            tranvse(root.right)
        tranvse(root)


    def next(self) -> int:
        return self.res.pop(0)


    def hasNext(self) -> bool:
        return len(self.res) != 0


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()
# @lc code=end

