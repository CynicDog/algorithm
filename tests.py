# Dependencies
import solutions

import sys
import re

import collections 
from typing import Optional

class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next


# Test Cases 
solution = solutions.Solution()

print(solution.isPalindrome("A man, a plan, a canal: Panama"))
print(solution.reverseString(['h', 'e', 'l', 'l', 'o']))
print(solution.mostCommonWord('Bob hit a ball, the hit BALL flew far after it was hit.', 'hit'))
print(solution.longestPalindrome('asdlkaasndssssslkkknaawww'))
print(solution.twoSum(nums = [3, 2, 4], target = 6))
print(solution.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
print(solution.threeSum(nums = [-1, 1, 3, -2, 2, -3]))
print(solution.productExceptSelf([1, 2, 3, 4, 5]))
print(solution.maxProfit([7, 1, 5, 3, 6, 4]))
print(solution.isLinkedListPalindrome(head = ListNode(1, ListNode(2, ListNode(3,None)))))
print(solution.isLinkedListPalindrome_2(head = ListNode(1, ListNode(2, ListNode(1,None)))))

merged = solution.mergedTwoLists(ListNode(1, ListNode(2, ListNode(4, None))), ListNode(3, ListNode(5, ListNode(6, None))))
while merged:
    print(merged.val, end =' ')
    merged = merged.next

print()

reversed = solution.reverseList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, None))))))
while reversed:
    print(reversed.val, end = ' ')
    reversed = reversed.next

print()

swapped = solution.swapPairs(ListNode(1, ListNode(2, ListNode(3, ListNode(4, None)))))
while swapped:
    print(swapped.val, end = ' ')
    swapped = swapped.next
