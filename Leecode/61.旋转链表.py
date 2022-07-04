#
# @lc app=leetcode.cn id=61 lang=python3
#
# [61] 旋转链表
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 1. 笨方法：从1推到n
        # slow = head
        # head_len = 0
        # while slow:
        #     slow = slow.next
        #     head_len += 1
        # if not head or not head.next or k%head_len == 0:
        #     return head
        # for i in range(k%head_len):
        #     if i != 0:
        #         head = dummy_node.next
        #     dummy_node = ListNode()
        #     slow = head
        #     fast = head.next
        #     while fast.next:
        #         slow = slow.next
        #         fast = fast.next
        #     slow.next = None
        #     fast.next = head
        #     dummy_node.next = fast
        # return dummy_node.next
        #
        # 2. 双指针
        slow = head
        head_len = 0
        while slow:
            slow = slow.next
            head_len += 1
        if not head or not head.next or k%head_len == 0:
            return head
        k = k % head_len
        slow = head
        fast = head
        for _ in range(k):
            fast = fast.next
        while fast.next:
            slow = slow.next
            fast = fast.next
        dummy_node = ListNode()
        tem = slow.next
        slow.next = None
        fast.next = head
        dummy_node.next = tem
        return dummy_node.next

# @lc code=end

