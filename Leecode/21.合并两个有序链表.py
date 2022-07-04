#
# @lc app=leetcode.cn id=21 lang=python3
#
# [21] 合并两个有序链表
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        left = list1
        right = list2
        tem = ListNode()
        dummy_node = ListNode(next=tem)
        while left and right:
            if left.val <= right.val:
                tem.next = left
                left = left.next
                tem = tem.next
            else:
                tem.next = right
                right = right.next
                tem = tem.next
        if left:
            tem.next = left
        if right:
            tem.next = right
        return dummy_node.next.next
# @lc code=end

