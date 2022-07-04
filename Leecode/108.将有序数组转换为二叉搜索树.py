#
# @lc app=leetcode.cn id=108 lang=python3
#
# [108] 将有序数组转换为二叉搜索树
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        # n = len(nums)
        # left = 0
        # right = n
        # def build_node(left, right):
        #     if left >= right:
        #         return
        #     mid = (left+right) // 2
        #     root = TreeNode(val=nums[mid])
        #     root.left = build_node(left, mid)
        #     root.right = build_node(mid+1, right)
        #     return root
        # return build_node(left, right)

        n = len(nums)
        left = 0
        right = n
        def build(left, right):
            if left >= right:
                return
            mid = (left+right) >> 1
            root = TreeNode(val=nums[mid])
            root.left = build(left, mid)
            root.right = build(mid, right)
            return root
        return build(left, right)
# @lc code=end

