#
# @lc app=leetcode.cn id=104 lang=python3
#
# [104] 二叉树的最大深度
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # 1. 层次遍历
        # if not root:
        #     return 0
        # node_list = [root]
        # k = 0
        # while node_list:
        #     k += 1
        #     for _ in range(len(node_list)):
        #         root = node_list.pop(0)
        #         if root.left:
        #             node_list.append(root.left)
        #         if root.right:
        #             node_list.append(root.right)
        # return k

        # 2. 深度遍历
        # self.depth = 0
        # self.res = 0
        # def tranverse(root):
        #     if not root:
        #         self.res = max(self.res, self.depth)
        #         return
        #     self.depth += 1
        #     tranverse(root.left)
        #     tranverse(root.right)
        #     self.depth -= 1
        # tranverse(root)
        # return self.res

        # 3. 后序遍历
        if not root:
            return 0
        left_max = self.maxDepth(root.left)
        right_max = self.maxDepth(root.right)
        return max(left_max, right_max) + 1
# @lc code=end

