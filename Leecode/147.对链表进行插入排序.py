#
# @lc app=leetcode.cn id=147 lang=python3
#
# [147] 对链表进行插入排序
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        # 转化为列表
        # res = [head]
        # head = head.next
        # while head:
        #     tem = []
        #     sort = False
        #     while not sort:
        #         if not res or res[-1].val <= head.val:
        #             sort = True
        #             tem.append(head)
        #         else:
        #             tem.append(res.pop())
        #     for node in tem[::-1]:
        #         res.append(node)
        #     head = head.next
        # for i in range(len(res)-1):
        #     res[i].next = res[i+1]
        # res[len(res)-1].next = None
        # return res[0]

        # 插入排序 n**2
        dummy_node = ListNode(next=head)
        curr = head.next
        last_node = head
        while curr:
            if last_node.val <= curr.val:
                # 说明原序列就是顺序不用动
                last_node = last_node.next
            else:
                prev = dummy_node
                while prev.next.val <= curr.val:
                    prev = prev.next
                last_node.next = curr.next
                curr.next = prev.next
                prev.next = curr
            curr = last_node.next
        return dummy_node.next

        # 归并排序 n*logn
        # def merge_sort(head):
        #     if not head or not head.next:
        #         return head
        #     fast = head
        #     slow = head
        #     while fast.next and fast.next.next:
        #         slow = slow.next
        #         fast = fast.next.next
        #     fast = slow
        #     slow = slow.next
        #     fast.next = None
        #     p = merge_sort(head)
        #     q = merge_sort(slow)
        #     return merge(p, q)

        # def merge(p, q):
        #     dummy_node = ListNode(-1)
        #     prev = dummy_node
        #     while p or q:
        #         val_1 = p.val if p else float("inf")
        #         val_2 = q.val if q else float("inf")
        #         if val_1 <= val_2:
        #             prev.next = p
        #             p = p.next
        #         else:
        #             prev.next = q
        #             q = q.next
        #         prev = prev.next
        #     return dummy_node.next
        # return merge_sort(head)                

# @lc code=end

