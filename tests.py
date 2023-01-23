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

# [LEETCODE #125. VALID PALINDROME]
print(solution.isPalindrome("A man, a plan, a canal: Panama"))

# [LEETCODE #344. REVERSE STRING]
print(solution.reverseString(['h', 'e', 'l', 'l', 'o']))

# [LEETCODE #819. MOST COMMON WORD]
print(solution.mostCommonWord('Bob hit a ball, the hit BALL flew far after it was hit.', 'hit'))

# [LEETCODE #5. LONGEST PALINDROME]
print(solution.longestPalindrome('asdlkaasndssssslkkknaawww'))

# [LEETCODE #1. TWO SUM] 
print(solution.twoSum(nums = [3, 2, 4], target = 6))

# [LEETCODE #42. TRAPPING RAIN WATER]
print(solution.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))

# [LEETCODE #15 THREE SUM]
print(solution.threeSum(nums = [-1, 1, 3, -2, 2, -3]))

# [LEETCODE #238 PRODUCT EXCEPT SELF]
print(solution.productExceptSelf([1, 2, 3, 4, 5]))

# [LEETCODE #121 MAX PROFIT(1)]
print(solution.maxProfit([7, 1, 5, 3, 6, 4]))

# [LEETCODE #234 PALINDROME LINKED LIST]
print(solution.isLinkedListPalindrome(head = ListNode(1, ListNode(2, ListNode(3,None)))))
print(solution.isLinkedListPalindrome_2(head = ListNode(1, ListNode(2, ListNode(1,None)))))

# [LEETCODE #21 MERGE TWO SORTED LISTS]
merged = solution.mergedTwoLists(ListNode(1, ListNode(2, ListNode(4, None))), ListNode(3, ListNode(5, ListNode(6, None))))
while merged:
    print(merged.val, end =' ')
    merged = merged.next

print()

# [LEETCODE #206 REVERSE LINKED LIST]
reversed = solution.reverseList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, None))))))
while reversed:
    print(reversed.val, end = ' ')
    reversed = reversed.next

print()

# [LEETCODE #24 SWAP NODES IN PAIRS]
swapped = solution.swapPairs(ListNode(1, ListNode(2, ListNode(3, ListNode(4, None)))))
while swapped:
    print(swapped.val, end = ' ')
    swapped = swapped.next

print() 

# [LEETCODE #328 ODD EVEN LINKED LIST]
odd_even = solution.oddEvenList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, ListNode(6, None)))))))
while odd_even:
	print(odd_even.val, end = ' ') 
	odd_even = odd_even.next 

print() 

# [LEETCODE #92 REVERSE LINKED LIST(2)]
reversed = solution.reverseBetween(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, ListNode(6, None)))))), left = 2, right = 5)
while reversed: 
	print(reversed.val, end = ' ') 
	reversed = reversed.next 

print()

# [LEETCODE #20 VALID PARENTHESES]
print(solution.isValidParentheses('()[]'))

# [LEETCODE #316 REMOVE DUPLICATE LETTERS]
print(solution.removeDuplicateLetters('bcabc'))

# [LEETCODE #739 DAILY TEMPERATURES]
print(solution.dailyTemperatures([73, 74, 75, 83, 79, 78, 77, 80]))
