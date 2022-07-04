#
# @lc app=leetcode.cn id=203 lang=python3
#
# [203] 移除链表元素
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy_node = ListNode()
        slow = dummy_node
        fast = head
        while fast:
            if fast.val != val:
                slow.next = fast
                slow = slow.next
            fast = fast.next
        slow.next = None
        return dummy_node.next
# @lc code=end

