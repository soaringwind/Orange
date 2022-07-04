#
# @lc app=leetcode.cn id=83 lang=python3
#
# [83] 删除排序链表中的重复元素
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # if not head:
        #     return
        # curr = head 
        # dummy_node = ListNode(next=curr)
        # while head:
        #     if curr.val != head.val:
        #         curr.next = head
        #         curr = head
        #     head = head.next
        # curr.next = None 
        # return dummy_node.next
        # dummy_node = ListNode()
        # curr = dummy_node
        # while head:
        #     if not head.next or head.val != head.next.val:
        #         curr.next = head
        #         curr = curr.next
        #     head = head.next
        # curr.next = None
        # return dummy_node.next
        if not head:
            return
        curr = head
        dummy_node = ListNode(next=curr)
        while head:
            if curr.val == head.val:
                head = head.next
            else:
                curr.next = head
                head = head.next
                curr = curr.next
        curr.next = None
        return dummy_node.next
# @lc code=end

