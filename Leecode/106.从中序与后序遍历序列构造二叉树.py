#
# @lc app=leetcode.cn id=106 lang=python3
#
# [106] 从中序与后序遍历序列构造二叉树
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # if len(inorder) == 0 and len(postorder) == 0:
        #     return 
        # root = TreeNode()
        # root_val = postorder[-1]
        # root.val = root_val
        # if len(inorder) == 1 and len(postorder) == 1:
        #     return root
        # root_left_inorder = inorder[:inorder.index(root_val)]
        # root_left_postorder = postorder[:len(root_left_inorder)]
        # root_right_inorder = inorder[inorder.index(root_val)+1:]
        # root_right_postorder = postorder[len(root_left_postorder):-1]
        # root.left = self.buildTree(root_left_inorder, root_left_postorder)
        # root.right = self.buildTree(root_right_inorder, root_right_postorder)
        # return root
        
        # 遇到反复查询则考虑使用哈希表
        # index_inorder_dic = {}
        # n = len(inorder)
        # for i in range(n):
        #     index_inorder_dic[inorder[i]] = i
        # def build(inorder, in_start, in_end, postorder, post_start, post_end):
        #     if in_end < in_start:
        #         return 
        #     root = TreeNode()
        #     root_val = postorder[post_end]
        #     root.val = root_val
        #     root_index = index_inorder_dic[root_val]
        #     left_size = root_index - in_start
        #     root.left = build(inorder, in_start, root_index-1, postorder, post_start, post_start+left_size-1)
        #     root.right = build(inorder, root_index+1, in_end, postorder, post_start+left_size, post_end-1)
        #     return root
        # return build(inorder, 0, n-1, postorder, 0, n-1)

        index_inorder_dic = {}
        n = len(inorder)
        for i in range(n):
            index_inorder_dic[inorder[i]] = i
        def build(inorder_start_index, inorder_end_index, postorder_start_index, postorder_end_index):
            if inorder_start_index > inorder_end_index:
                return
            root_val = postorder[postorder_end_index]
            root = TreeNode()
            root.val = root_val
            inorder_index = index_inorder_dic[root_val]
            # print(inorder_index)
            left_size = inorder_index - inorder_start_index
            root.left = build(inorder_start_index, inorder_index-1, postorder_start_index, postorder_start_index+left_size-1)
            root.right = build(inorder_index+1, inorder_end_index, postorder_start_index+left_size, postorder_end_index-1)
            return root
        return build(0, n-1, 0, n-1)

# @lc code=end

