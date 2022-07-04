#
# @lc app=leetcode.cn id=15 lang=python3
#
# [15] 三数之和
#

# @lc code=start
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n):
            a = nums[i]
            left = i+1
            right = n-1
            if i > 0 and nums[i-1] == a:
                continue
            while left < right:
                b = nums[left]
                c = nums[right]
                if a+b+c > 0:
                    right -= 1
                elif a+b+c < 0:
                    left += 1
                else:
                    res.append([a, b, c])
                    while left < right and nums[left] == b:
                        left += 1
                    while left < right and nums[right] == c:
                        right -= 1
        return res
# @lc code=end

