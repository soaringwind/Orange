#
# @lc app=leetcode.cn id=112 lang=python3
#
# [112] 路径总和
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        self.path_sum = 0
        res = []
        def tranvse(root):
            if not root:
                return 
            self.path_sum += root.val
            flag_1 = tranvse(root.left)
            flag_2 = tranvse(root.right)
            if not flag_1 and not flag_2:
                if self.path_sum == targetSum:
                    res.append(self.path_sum)
            self.path_sum -= root.val
            return root
        tranvse(root)
        return len(res) != 0
# @lc code=end

