#
# @lc app=leetcode.cn id=27 lang=python3
#
# [27] 移除元素
#

# @lc code=start
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        nums.sort()
        left = 0
        n = len(nums)
        ele_len = 0
        right = n-1
        while left < n:
            if nums[left] == val:
                nums[left] = nums[right]
                right -= 1
                ele_len += 1
            left += 1
        for i in range(ele_len):
            nums.pop()
# @lc code=end

