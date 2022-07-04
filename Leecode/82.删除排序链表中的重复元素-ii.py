#
# @lc app=leetcode.cn id=82 lang=python3
#
# [82] 删除排序链表中的重复元素 II
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return
        dummy_node = ListNode()
        tem = dummy_node
        left = head
        right = head.next
        while right:
            flag = False
            while right and left.val == right.val:
                right = right.next
                flag = True
            if not flag:
                tem.next = left
                tem = tem.next
            if right:
                left = right
                right = right.next
        if not left.next:
            tem.next = left
            tem = tem.next
        tem.next = None
        return dummy_node.next
# @lc code=end

