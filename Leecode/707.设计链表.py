#
# @lc app=leetcode.cn id=707 lang=python3
#
# [707] 设计链表
#

# @lc code=start
class MyLinkedList:

    def __init__(self):
        self.val = []


    def get(self, index: int) -> int:
        if index >= len(self.val):
            return -1
        else:
            return self.val[index]


    def addAtHead(self, val: int) -> None:
        self.val.insert(0, val)


    def addAtTail(self, val: int) -> None:
        self.val.append(val)


    def addAtIndex(self, index: int, val: int) -> None:
        if index == len(self.val):
            self.val.append(val)
            return
        if index > len(self.val):
            return
        if index < 0:
            self.val.insert(0, val)
            return
        self.val.insert(index, val)


    def deleteAtIndex(self, index: int) -> None:
        if index < len(self.val):
            self.val.pop(index)



# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
# @lc code=end

