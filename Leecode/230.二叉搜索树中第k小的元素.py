#
# @lc app=leetcode.cn id=230 lang=python3
#
# [230] 二叉搜索树中第K小的元素
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # 中序遍历
        # node_list = [] # stack FIFO
        # val_list = []
        # node = root
        # while True:
        #     while node:
        #         node_list.append(node)
        #         node = node.left
        #     if not node_list:
        #         break
        #     visit_node = node_list.pop()
        #     val_list.append(visit_node.val)
        #     node = visit_node.right
        # return val_list[k-1]

        # 递归
        val = []
        def tranverse(root):
            if not root:
                return
            if tranverse(root.left):
                return val
            val.append(root.val)
            if len(val) == k:
                return val
            if tranverse(root.right):
                return val
        return tranverse(root)[-1]
# @lc code=end

