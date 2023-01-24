# Dependencies
import sys 
import re 
import collections 
import heapq 
import random 

from typing import Optional, List 

class ListNode:
	def __init__(self, val = 0, next = None):
		self.val = val
		self.next = next

# Soultions 
class Solution: 

# [LEETCODE #125. VALID PALINDROME]
	def isPalindrome(self, s: str) -> bool: 
		
		strs = [] 

		for char in s: 
			if char.isalnum(): 
				strs.append(char.lower()) 

		while len(strs) > 1: 
			if strs.pop(0) != strs.pop(): 
				return False 

		return True 


# [LEETCODE #344. REVERSE STRING]
	def reverseString(self, s: list[str]) -> list[str]: 
		
		left, right = 0, len(s) - 1 
		
		while left < right: 
			s[left], s[right] = s[right], s[left]
			
			left += 1 
			right -= 1  

		return s 


# [LEETCODE #819. MOST COMMON WORD]
	def mostCommonWord(self, paragraph: str, banned: list[str]) -> str: 

		words = [word for word in re.sub(r'[^\w]', ' ', paragraph).lower().split() if word not in banned] 

		counts = collections.Counter(words) 

		return counts.most_common(1)[0][0]


# [LEETCODE #49. GROUP ANAGRAM]
	def groupAnagram(self, strs: list[str]) -> list[list[str]]: 
		
		anagrams = collections.defaultdict(list)
	
		for word in strs: 
			anagrams[''.join(sorted(word))].append(word) 

		return list(anagrams.values())


# [LEETCODE #5. LONGEST PALINDROME] 
	def longestPalindrome(self, s: str) -> str: 
		
		def expand(left: int, right: int) -> str: 
			
			while left >= 0 and right < len(s) and s[left] == s[right]: 

				left -= 1  
				right += 1 

			return s[left + 1 : right] 

		if len(s) < 2 or s == s[::-1]:

			return s

		result = '' 
		for i in range(len(s) - 1): 
			result = max(result, expand(i, i + 1), expand(i, i + 2), key = len) 

		return result 


# [LEETCODE #1. TWO SUM] 
	def twoSum(self, nums: list[int], target: int) -> list[int]: 

		numsMap = {}

		for idx, num in enumerate(nums): 
			if target - num in numsMap: 
				return [idx, numsMap[target - num]]

			numsMap[num] = idx 


# [LEETCODE #42. TRAPPING RAIN WATER]
	def trap(self, height: list[int]) -> int: 

		if not height: 
			return 0 

		volume = 0 
		left, right = 0, len(height) - 1 

		left_max, right_max = height[left], height[right] 

		while left < right:
			left_max, right_max = max(left_max, height[left]), max(right_max, height[right])
			
			if left_max <= right_max: 
				volume += left_max - height[left] 
				left += 1 

			else:
				volume += right_max - height[right] 
				right -= 1 

		return volume 


# [LEETCODE #15 THREE SUM] 
	def threeSum(self, nums: list[int]) -> list[list[int]]: 

		results = [] 

		nums.sort() 

		for i in range(len(nums) - 2): 
			if i > 0 and nums[i] == nums[i - 1]:
				continue 

			left, right = i + 1, len(nums) - 1 

			while left < right: 
				sum = nums[i] + nums[left] + nums[right] 

				if sum < 0: 
					left += 1 

				elif sum > 0:
					right -= 1 

				else: 
					results.append([nums[i], nums[left], nums[right]])

					while left < right and nums[left] == nums[left + 1]:
						left += 1 

					while left < right and nums[right] == nums[right - 1]:
						right -= 1 
					
					left += 1 
					right -= 1 

		return results 


# [LEETCODE #238 PRODUCT EXCEPT SELF] 
	def productExceptSelf(self, nums: list[int]) -> int:

		out = [] 
	
		p = 1 
		for i in range(0, len(nums)): 
			out.append(p) 
			p = p * nums[i] 

		p = 1 
		for i in range(len(nums) - 1, - 1, -1): 
			out[i] = out[i] * p 
			p = p * nums[i] 

		return out 


# [LEETCODE #121 MAX PROFIT(1)]
	def maxProfit(self, prices: list[int]) -> int: 

		profit = 0 
		min_price = sys.maxsize 

		for price in prices: 
			min_price = min(min_price, price) 
			profit = max(profit, price - min_price) 

		return profit 


# [LEETCODE #234 PALINDROME LINKED LIST] 
	def isLinkedListPalindrome(self, head: Optional[ListNode]) -> bool: 

		deq: Deque = collections.deque() 
        
		if not head: 
			return True 
        
		node = head 
        
		while node: 
			deq.append(node.val) 
			node = node.next 
		            
		while len(deq) > 1: 
			if deq.popleft() != deq.pop(): 
				return False 

		
		return True 				

	def isLinkedListPalindrome_2(self, head: Optional[ListNode]) -> bool: 
		
		rev = None 
		slow = fast = head 

		while fast and fast.next: 
			fast = fast.next.next
			rev, rev.next, slow = slow, rev, slow.next

		if fast:
			slow = slow.next 

		while rev and rev.val == slow.val: 
			slow, rev = slow.next, rev.next 

		return not rev 

	
# [LEETCODE #21 MERGE TWO SORTED LISTS] 
	def mergedTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]: 
		if not list1 or (list2 and list1.val > list2.val): 
			list1, list2 = list2, list1 

		if list1: 
			list1.next = self.mergedTwoLists(list1.next, list2) 

		return list1 


# [LEETCODE #206 REVERSE LINKED LIST(1)] 
	def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]: 

		node, prev = head, None 

		while node: 
			next, node.next = node.next, prev 
			prev, node = node, next 

		return prev 


# [LEETCODE #24 SWAP NODES IN PAIRS] 
	def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]: 
		
		cur = head 

		while cur and cur.next: 
			cur.val, cur.next.val = cur.next.val, cur.val 
			cur = cur.next.next 

		return head  


# [LEETCODE #328 ODD EVEN LINKED LIST]
	def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]: 
	
		if head is None: 
			return None 

		odd = head 
		odd_head = head 

		even = head.next 
		even_head = head.next 

		while even and even.next: 
			odd.next, even.next = odd.next.next, even.next.next 
			odd, even = odd.next, even.next 

		odd.next = even_head 		

		return odd_head 


# [LEETCODE #92 REVERSE LINKED LIST(2)]
	def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]: 
		
		if not head or left == right: 
			return head 

		root = start = ListNode(None) 
		root.next = head 

		for _ in range(left - 1): 
			start = start.next 

		end = start.next 

		for _ in range(right - left): 
			tmp = start.next 
			start.next = end.next 
			end.next = end.next.next
			start.next.next = tmp 

		return root.next 


# [LEETCODE #20 VALID PARENTHESES] 
	def isValidParentheses(self, s: str) -> bool: 

		stack = [] 
		table = {
			')': '(', 
			'}': '{', 
			']': '[' 
		}		

		for char in s: 
			if char not in table:
				stack.append(char) 

			elif not stack or stack.pop() != table[char]:
				return False 

		return len(stack) == 0 


# [LEETCODE #316 REMOVE DUPLICATE LETTERS] 
	def removeDuplicateLetters(self, s: str) -> str: 

		seen, stack, counter = set(), [], collections.Counter(s) 
		
		for char in s: 
			counter[char] -= 1 
	
			if char in seen: 
				continue 

			while stack and char < stack[-1] and counter[stack[-1]] > 0: 
				seen.remove(stack.pop())

			seen.add(char)
			stack.append(char) 

		return ''.join(stack) 


# [LEETCODE #739 DAILY TEMPERATURES] 
	def dailyTemperatures(self, T: List[int]) -> List[int]: 

		answer = [0] * len(T) 
		stack = [] 

		for i, cur in enumerate(T): 

			while stack and cur > T[stack[-1]]: 
				last = stack.pop()
				answer[last] = i - last 

			stack.append(i) 

		return answer 
		
		
# [LEETCODE #23 MERGE K SORTED LISTS]
	def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]: 
		
		root = result = ListNode(None) 
		heap = [] 

		for i in range(len(lists)): 
			if lists[i]: 
				heapq.heappush(heap, (lists[i].val, i, lists[i])) 

		while heap: 
			node = heapq.heappop(heap) 
			idx = node[1] 
			result.next = node[2] 

			result = result.next 	
			
			if result.next: 
				heapq.heappush(heap, (result.next.val, idx, result.next))

		return root.next  


# [BIRTHDAY PROBLEM]
	def birthdayProblem(self): 
		trials = 10000 
		count = 0 

		for _ in range(trials): 
			birthdays = [] 

			for _ in range(23): 
				birthday = random.randint(1, 365)
	
				if birthday in birthdays: 
					count += 1 
					break; 

				birthdays.append(birthday) 
		
		print(f'{count / trials}')
		
			
