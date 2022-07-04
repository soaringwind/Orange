#
# @lc app=leetcode.cn id=148 lang=python3
#
# [148] 排序链表
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next



class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 
        dummy_node = ListNode()
        
        return


        # 哈希表/递归
        # if not head:
        #     return head
        # curr = head 
        # res = {}
        # while curr:
        #     res[curr] = curr.val
        #     curr = curr.next
        # res = sorted(res.items(), key=lambda x:x[1])
        # curr = res[0][0]
        # dummy_node = ListNode(next=curr)
        # for k, v in res[1:]:
        #     curr.next = k
        #     curr = curr.next
        # curr.next = None
        # return dummy_node.next
        # 归并排序
        # if not head:
        #     return
        # tem = head
        # tem_list = []
        # n = 0
        # while tem:
        #     n += 1
        #     tem_list.append(tem)
        #     tem = tem.next
        # def sort_node(left, right):
        #     if right - left < 2:
        #         tem_ = tem_list[left]
        #         tem_.next = None
        #         return tem_
        #     mid = (left + right) // 2
        #     left_node = sort_node(left, mid)
        #     right_node = sort_node(mid, right)
        #     dummy_node = ListNode()
        #     if left_node.val < right_node.val:
        #         curr = left_node
        #         left_node = left_node.next
        #     else:
        #         curr = right_node
        #         right_node = right_node.next
        #     dummy_node.next = curr
        #     while left_node or right_node:
        #         if left_node and (not right_node or left_node.val < right_node.val):
        #             curr.next = left_node
        #             left_node = left_node.next
        #         else:
        #             curr.next = right_node
        #             right_node = right_node.next
        #         curr = curr.next
        #     curr.next = None
        #     return dummy_node.next
        # return sort_node(0, n)
# @lc code=end

