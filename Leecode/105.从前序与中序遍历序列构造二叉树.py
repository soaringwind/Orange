#
# @lc app=leetcode.cn id=105 lang=python3
#
# [105] 从前序与中序遍历序列构造二叉树
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        inorder_index_dic = {}
        n = len(inorder)
        for i in range(n):
            inorder_index_dic[inorder[i]] = i
        def build(preorder, pre_start, pre_end, inorder, in_start, in_end):
            if pre_end <= pre_start:
                return 
            root = TreeNode()
            root.val = preorder[pre_start]
            if pre_end - pre_start < 2:
                return root
            root_index = inorder_index_dic[root.val]
            left_len = root_index - in_start
            # right_len = in_end - root_index
            root.left = build(preorder, pre_start+1, pre_start+1+left_len, inorder, in_start, root_index)
            root.right = build(preorder, pre_start+1+left_len, pre_end, inorder, root_index+1, in_end)
            return root
        return build(preorder, 0, n, inorder, 0, n)
# @lc code=end

