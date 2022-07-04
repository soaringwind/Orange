#
# @lc app=leetcode.cn id=101 lang=python3
#
# [101] 对称二叉树
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right




class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        # 从左到右判断是否相同（层次遍历）FIFO队列/递归
        # if not root:
        #     return False
        # node_list = [root]
        # while node_list:
        #     # 为了直接对比是否对称，因此每一次遍历用val来储存值
        #     tem = node_list # 为了每次遍历完一层再进入下一层
        #     node_list = []
        #     val = []
        #     while tem:
        #         node = tem.pop(0)
        #         if node:
        #             node_list.append(node.left)
        #             node_list.append(node.right)
        #             val.append(node.val)
        #         else:
        #             val.append(None)
        #     if val != val[::-1]:
        #         return False
        # return True

        # def func(left, right):
        #     if left == right:
        #         return True
        #     if left == None or right == None or left.val != right.val:
        #         return False
        #     return func(left.left, right.right) and func(left.right, right.left)
        # return func(root.left, root.right)

        # 层次遍历 for + while
        if not root:
            return 
        node_list = [root]
        while node_list:
            val_list = []
            for _ in range(len(node_list)):
                node = node_list.pop(0)
                if node:
                    node_list.append(node.left)
                    node_list.append(node.right)
                    val_list.append(node.val)
                else:
                    val_list.append(None)
            if val_list[::-1] != val_list:
                return False
        return True
# @lc code=end

