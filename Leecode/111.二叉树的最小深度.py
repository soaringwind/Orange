#
# @lc app=leetcode.cn id=111 lang=python3
#
# [111] 二叉树的最小深度
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        # 1. 深度遍历
        # if not root:
        #     return 0
        # self.depth = 0
        # self.res = float("inf")
        # def tranverse(root):
        #     if not root:
        #         return
        #     self.depth += 1
        #     flag_1 = tranverse(root.left)
        #     flag_2 = tranverse(root.right)
        #     if not flag_1 and not flag_2:
        #         self.res = min(self.res, self.depth)
        #     self.depth -= 1
        #     return self.res
        # return tranverse(root)

        # 2. 层次遍历
        if not root:
            return 0
        node_list = []
        node_list.append(root)
        k = 0
        while node_list:
            k += 1
            for _ in range(len(node_list)):
                root = node_list.pop(0)
                leaf = True
                if root.left:
                    node_list.append(root.left)
                    leaf = False
                if root.right:
                    node_list.append(root.right)
                    leaf = False
                if leaf:
                    return k
# @lc code=end

