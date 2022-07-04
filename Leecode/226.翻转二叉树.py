#
# @lc app=leetcode.cn id=226 lang=python3
#
# [226] 翻转二叉树
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return
        def inverse(node, left, right):
            node.right = left
            node.left = right
            return node
        res = []
        res.append(root)
        while res:
            node = res.pop()
            if node.left:
                res.append(node.left)
            if node.right:
                res.append(node.right)
            inverse(node, node.left, node.right)
        return root
# @lc code=end

