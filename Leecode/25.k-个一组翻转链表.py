#
# @lc app=leetcode.cn id=25 lang=python3
#
# [25] K 个一组翻转链表
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # dummy_node = ListNode()
        # tem = []
        # prev = dummy_node
        # while head:
        #     tem.append(head)
        #     head = head.next
        #     if len(tem) == k:
        #         while tem:
        #             prev.next = tem.pop()
        #             prev = prev.next
        # while tem:
        #     prev.next = tem.pop(0)
        #     prev = prev.next
        # prev.next = None
        # return dummy_node.next

        dummy_node = ListNode()
        prev = dummy_node
        slow = head
        fast = head
        num = 0
        while fast:
            if num == k:
                while num > 0:
                    cur = slow
                    slow = slow.next
                    tem = prev.next
                    prev.next = cur
                    prev.next.next = tem
                    num -= 1
                for _ in range(k):
                    prev = prev.next
            fast = fast.next
            num += 1
        if num == k:
            while num > 0:
                cur = slow
                slow = slow.next
                tem = prev.next
                prev.next = cur
                prev.next.next = tem
                num -= 1
        else:
            prev.next = slow
        return dummy_node.next
# @lc code=end

