#
# @lc app=leetcode.cn id=206 lang=python3
#
# [206] 反转链表
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next




class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # # 双指针
        # curr = head 
        # prev = None
        # while curr:
        #     tem = curr.next
        #     curr.next = prev
        #     prev = curr
        #     curr = tem
        # return prev
        dummy_node = ListNode()
        while head:
            tem = head.next
            head.next = dummy_node.next
            dummy_node.next = head
            head = tem
        return dummy_node.next
# @lc code=end

