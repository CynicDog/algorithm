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

class TreeNode: 
    def __init__(self, val = 0, left = None, right = None): 
        self.val = val
        self.left = left  
        self.right = right 

# Test Cases 
solution = solutions.Solution()

# [LEETCODE #125. VALID PALINDROME]
print("#125\t", solution.isPalindrome("A man, a plan, a canal: Panama"))

# [LEETCODE #344. REVERSE STRING]
print("#344\t", solution.reverseString(['h', 'e', 'l', 'l', 'o']))

# [LEETCODE #819. MOST COMMON WORD]
print("#819\t", solution.mostCommonWord('Bob hit a ball, the hit BALL flew far after it was hit.', 'hit'))

# [LEETCODE #5. LONGEST PALINDROME]
print("#5\t", solution.longestPalindrome('asdlkaasndssssslkkknaawww'))

# [LEETCODE #1. TWO SUM] 
print("#1\t", solution.twoSum(nums = [3, 2, 4], target = 6))

# [LEETCODE #42. TRAPPING RAIN WATER]
print("#42\t", solution.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))

# [LEETCODE #15 THREE SUM]
print("#15\t", solution.threeSum(nums = [-1, 1, 3, -2, 2, -3]))

# [LEETCODE #238 PRODUCT EXCEPT SELF]
print("#238\t", solution.productExceptSelf([1, 2, 3, 4, 5]))

# [LEETCODE #121 MAX PROFIT(1)]
print("#121\t", solution.maxProfit([7, 1, 5, 3, 6, 4]))

# [LEETCODE #234 PALINDROME LINKED LIST]
print("#234\t", solution.isLinkedListPalindrome(head = ListNode(1, ListNode(2, ListNode(3,None)))))
print("#234\t", solution.isLinkedListPalindrome_2(head = ListNode(1, ListNode(2, ListNode(1,None)))))

# [LEETCODE #21 MERGE TWO SORTED LISTS]
merged = solution.mergedTwoLists(ListNode(1, ListNode(2, ListNode(4, None))), ListNode(3, ListNode(5, ListNode(6, None))))
print("#21\t", end = ' ')
while merged:
    print(merged.val, end =' ')
    merged = merged.next

print()

# [LEETCODE #206 REVERSE LINKED LIST]
reversed = solution.reverseList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, None))))))
print("#206\t", end = ' ')
while reversed:
    print(reversed.val, end = ' ')
    reversed = reversed.next

print()

# [LEETCODE #24 SWAP NODES IN PAIRS]
swapped = solution.swapPairs(ListNode(1, ListNode(2, ListNode(3, ListNode(4, None)))))
print("#24\t", end = ' ')
while swapped:
    print(swapped.val, end = ' ')
    swapped = swapped.next

print() 

# [LEETCODE #328 ODD EVEN LINKED LIST]
odd_even = solution.oddEvenList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, ListNode(6, None)))))))
print("#328\t", end = ' ')
while odd_even:
	print(odd_even.val, end = ' ') 
	odd_even = odd_even.next 

print() 

# [LEETCODE #92 REVERSE LINKED LIST(2)]
reversed = solution.reverseBetween(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, ListNode(6, None)))))), left = 2, right = 5)
print("#92\t", end = ' ')
while reversed: 
	print(reversed.val, end = ' ') 
	reversed = reversed.next 

print()

# [LEETCODE #20 VALID PARENTHESES]
print("#20\t", solution.isValidParentheses('()[]'))

# [LEETCODE #316 REMOVE DUPLICATE LETTERS]
print("#316\t", solution.removeDuplicateLetters('bcabc'))

# [LEETCODE #739 DAILY TEMPERATURES]
print("#739\t", solution.dailyTemperatures([73, 74, 75, 83, 79, 78, 77, 80]))

# [LEETCODE #23 MERGE K SORTED LISTS]

mergedKLists = solution.mergeKLists([ListNode(1, ListNode(4, ListNode(5, None))), ListNode(1, ListNode(3, ListNode(4, None))), ListNode(2, ListNode(6, None))])
print("#23\t", end =' ')
while mergedKLists: 
	print(mergedKLists.val, end = ' ') 
	mergedKLists = mergedKLists.next

print() 

# [BIRTHDAY PROBLEM]
print("Birthday Problem\t", end = ' ')
solution.birthdayProblem()

# [LEETCODE #771 JEWELS AND STONES]
print("#771\t", solution.numJewelsInStones(jewels = "aA", stones = "aAAbbBBbb")) 

# [LEETCODE #3 LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS]
print("#3\t", solution.lengthOfLongestSubstring("aabbcabdbddbc")) 

# [LEETCODE #347 TOP K FREQUENT ELEMENTS]
print("#347\t", solution.topKFrequent([1, 1, 1, 2, 2, 3], k = 2))

# [LEETCODE #200 NUMBER OF ISLANDS]  
print("#200\t", solution.numIslands(grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","1"],
  ["0","0","1","0","1"]]))

# [LEETCODE #17 LETTER COMBINATIONS OF A PHONE NUMBER] 
print("#17\t", solution.letterCombinations("23"))

# [LEETCODE #46 PERMUTATION]
print("#46\t", solution.permute([1, 2, 3]))

# [LEETCODE #77 COMBINATION] 
print("#77\t", solution.combine(4, 2))

# [LEETCODE #39 COMBINATION SUM] 
print("#39\t", solution.combinationSum(candidates = [2,3,6,7], target = 7))

# [LEETCODE #78 SUBSETS]
print("#78\t", solution.subsets(nums = [1, 2, 5]))

# [LEETCODE #332 RECONSCRUCT ITINERARY] 
print("#332\t", solution.findItinerary([["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]))

# [LEETCODE #207 COURSE SCHEDULE]
print("#207\t", solution.canFinish(5, prerequisites = [[1,4], [2,4], [3, 1], [3, 2]]))

# [LEETCODE #743 NETWORK DELAY TIME]
print("#743\t", solution.networkDelayTime(times = [[2, 1, 1], [2, 6, 2], [2, 3, 2], [1, 3, 3], [6, 5, 2], [5, 3, 3], [3, 4, 4], [3, 7, 3]],n = 7, k = 2))

# [LEETCODE #787 CHEAPEST FLIGHTS WITHIN K STOPS] 
print("#787\t", solution.findCheapestPrice(n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1)) 
print("#787\t", solution.findCheapestPrice(n = 4, flights = [[0,1,1], [1,2,1], [0,2,5],[2,3,1]], src = 0, dst = 3, k = 1))

# [LEETCODE #104 MAXIMUM DEPTH OF BINARY TREE] 
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15) 
root.right.left.left = TreeNode(30)
root.right.left.right = TreeNode(45)
root.right.left.right.right = TreeNode(60)
root.right.right = TreeNode(7)

print("#104\t", solution.maxDepth(root))
print("#104\t", solution.maxDepthRecursive(root))

# [LEETCODE #543 DIAMETER OF BINARY TREE] 
root = TreeNode(1)
root.left = TreeNode(2)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right = TreeNode(3)

print("#543\t", solution.diameterOfBinaryTree(root))

# [LEETCODE #687 LONGEST UNIQUE VALUE PATH]
root = TreeNode(1)
root.right = TreeNode(1) 
root.right.left = TreeNode(1) 
root.right.left.left = TreeNode(1) 
root.right.left.right = TreeNode(1) 
root.right.right = TreeNode(1)
root.right.right.left = TreeNode(1) 

print("#687\t", solution.longestUnivaluePath(root)) 

# [LEETCODE #226 INVERT BINARY TREE]
root = TreeNode(4) 
root.left = TreeNode(2) 
root.left.left = TreeNode(1)
root.left.right = TreeNode(3) 
root.right = TreeNode(7) 
root.right.left = TreeNode(6) 
root.right.right = TreeNode(9) 

inverted = solution.invertTree(root)

queue = collections.deque([inverted]) 
print("#226\t", end = ' ')
while queue: 

	for _ in range(len(queue)): 
		node = queue.popleft() 
		print(node.val, end = ' ') 

		if node.left:
			queue.append(node.left)
		if node.right:
			queue.append(node.right) 

print() 

inverted_while = solution.invertTreeWhile(root)

queue = collections.deque([inverted_while])   
print("#226\t", end = ' ')
while queue: 

    for _ in range(len(queue)):
        node = queue.popleft()
        print(node.val, end = ' ')

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
print() 

# [LEETCODE #617 MERGE TWO BINARY TREES] 
root1 = TreeNode(1)
root1.left = TreeNode(3)
root1.left.left = TreeNode(5) 
root1.right = TreeNode(2) 

root2 = TreeNode(2)
root2.left = TreeNode(1)
root2.left.right = TreeNode(4) 
root2.right = TreeNode(3) 
root2.right.right = TreeNode(7) 

merged = solution.mergeTrees(root1, root2)

queue = collections.deque([merged]) 
print("#617\t", end = ' ')
while queue:

    for _ in range(len(queue)):
        node = queue.popleft()
        print(node.val, end = ' ')

        if node.left: 
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
print()

# [LEETCODE #110 BALANCED BINARY TREE] 
root = TreeNode(1)
root.left = TreeNode(2)
root.left.left = TreeNode(3)
root.left.left.left = TreeNode(4)
root.right = TreeNode(2) 
root.right.right = TreeNode(3)
root.right.right.right = TreeNode(4) 

print("#110\t", solution.isBalanced(root))

# [LEETCODE #310 MINIMUN HEIGHT TREES] 
print("#310\t", solution.findMinHeightTrees(n = 8, edges = [[3, 0], [3, 1], [3, 2], [3, 4], [3, 6], [4, 5], [6, 7]]))

# [LEETCODE #108 CONVERT SORTED ARRAY TO BINARY SEARCH TREE]  
BST = solution.sortedArrayToBST([-10, -3, 0, 5, 9]) 
print("#108\t", end = ' ') 
queue = collections.deque([BST]) 
while queue: 
	for _ in range(len(queue)): 
		node = queue.popleft() 
		print(node.val, end = ' ') 

		if node.left:
			queue.append(node.left) 
		if node.right:
			queue.append(node.right)  
print() 

# [LEETCODE #1038 BINARY SEARCH TREE TO GREATER SUM TREE]
root = TreeNode(4) 
root.left = TreeNode(1) 
root.left.left = TreeNode(0) 
root.left.right = TreeNode(2) 
root.left.right.right = TreeNode(3) 
root.right = TreeNode(6) 
root.right.left = TreeNode(5) 
root.right.right = TreeNode(7) 
root.right.right.right = TreeNode(8) 

converted = solution.bstToGst(root) 
print("#1038\t", end = ' ')
queue = collections.deque([converted]) 
while queue: 
	for _ in range(len(queue)): 
		node = queue.popleft()
		print(node.val, end = ' ') 
		
		if node.left:
			queue.append(node.left) 
		if node.right:
			queue.append(node.right) 
print()


# [LEETCODE #938 RANGE SUM OF BST] 
root = TreeNode(10) 
root.left = TreeNode(5) 
root.left.left = TreeNode(3) 
root.left.right = TreeNode(7) 
root.right = TreeNode(15) 
root.right.right = TreeNode(18) 

print("#938\t", solution.rangeSumBST(root, low = 7, high = 15)) 

# [LEETCODE #783 MINIMUM DISTANCE BETWEEN BST NODES] 
root = TreeNode(90) 
root.left = TreeNode(69) 
root.left.left = TreeNode(49) 
root.left.left.right = TreeNode(52) 
root.left.right = TreeNode(89) 

print("#783\t", solution.minDiffInBST(root))

# [LEETCODE #336 PALINDROME PAIRS] 
print("#336\t", solution.palindromePairs(["abcd","dcba","lls","s","sssll", "cba"])) 
