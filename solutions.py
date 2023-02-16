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

class TreeNode: 
    def __init__(self, val = 0, left = None, right = None): 
        self.val = val
        self.left = left 
        self.right = right 

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
		
			
# [LEETCODE #771 JEWELS AND STONES]
	def numJewelsInStones(self, jewels: str, stones: str) -> int: 

		freq = {}
		count = 0 

		for stone in stones: 
			if stone not in freq: 
				freq[stone] = 1 
			else:
				freq[stone] += 1 

		for jewel in jewels: 
			if jewel in freq: 
				count += freq[jewel]  

		return count 


# [LEETCODE #3 LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS] 
	def lengthOfLongestSubstring(self, s: str) -> int:
		
		start = max_length = 0 
		seen = {} 

		for idx, char in enumerate(s): 
			if char in seen and start <= seen[char]: 
				start = seen[char] + 1 
			else: 
				max_length = max(max_length, idx - start + 1) 

			seen[char] = idx 

		return max_length


# [LEETCODE #347 TOP K FREQUENT ELEMENTS]
	def topKFrequent(self, nums: List[int], k: int) -> List[int]:
		
		freqs = collections.Counter(nums) 
		heap = [] 

		for num in freqs: 
			heapq.heappush(heap, (-freqs[num], num))

		result = []
		for _ in range(k):
			result.append(heapq.heappop(heap)[1])

		return result 

# [LEETCODE #200 NUMBER OF ISLANDS] 
	def numIslands(self, grid: List[List[str]]) -> int: 
		
		def dfs(i, j):
			if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1': 
				return 

			grid[i][j] = 0 

			dfs(i + 1, j)
			dfs(i - 1, j) 
			dfs(i, j + 1)
			dfs(i, j - 1) 

		count = 0 
		for i in range(len(grid)):
			for j in range(len(grid[0])): 
				if grid[i][j] == '1':
					dfs(i, j) 
					count += 1 

		return count 

# [LEETCODE #17 LETTER COMBINATIONS OF A PHONE NUMBER] 
	def letterCombinations(self, digits: str) -> List[str]: 

		dic = {
			"2": "abc", 
			"3": "def", 
			"4": "ghi", 
			"5": "jkl", 
			"6": "mno", 
			"7": "pqrs", 
			"8": "tuv", 
			"9": "wxyz"
		}

		result = [] 

		def dfs(index, path): 
			if len(path) == len(digits): 
				result.append(path)
				return 

			for i in range(index, len(digits)): 
				for j in dic[digits[i]]: 
					dfs(i + 1, path + j) 

		dfs(0, "")

		return result 			
		
# [LEETCODE #46 PERMUTATION]
	def permute(self, nums: List[int]) -> List[List[int]]: 
		
		result = []
		prev_elem = [] 

		def dfs(elements): 
			if len(elements) == 0: 
				result.append(prev_elem[:]) 
				return 

			for elem in elements: 
				next_elem = elements[:] 
				next_elem.remove(elem)

				prev_elem.append(elem) 
	
				dfs(next_elem) 

				prev_elem.pop() 

		dfs(nums) 

		return result 


# [LEETCODE #77 COMBINATION] 
	def combine(self, n: int, k : int) -> List[List[int]]: 

		result = [] 

		def dfs(elements, start, k): 
			if k == 0:
				result.append(elements[:]) 

			for i in range(start, n + 1): 
				elements.append(i) 

				dfs(elements, i + 1, k - 1) 

				elements.pop() 

		dfs([], 1, k) 

		return result 


# [LEETCODE #39 COMBINATION SUM]
	def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]: 

		result = [] 

		def dfs(csum, index, path): 
			if csum < 0: 
				return 

			if csum == 0: 
				result.append(path)
				return 

			for i in range(index, len(candidates)): 
				dfs(csum - candidates[i], i, path + [candidates[i]]) 

		dfs(target, 0, [])

		return result


# [LEETCODE #78 SUBSETS]
	def subsets(self, nums: List[int]) -> List[List[int]]:

		result = [] 

		def dfs(index, path): 
			result.append(path)

			for i in range(index, len(nums)): 
				dfs(i + 1, path + [nums[i]])  

		dfs(0, [])
	
		return result  


# [LEETCODE #332 RECONSCRUCT ITINERARY] 
	def findItinerary(self, tickets: List[List[str]]) -> List[str]:
		
		graph = collections.defaultdict(list) 

		for x, y in sorted(tickets): 
			graph[x].append(y) 

		result = []
		def dfs(a): 
			while graph[a]:
				dfs(graph[a].pop(0)) 
			
			result.append(a)

		dfs("JFK")

		return result[::-1]


# [LEETCODE #207 COURSE SCHEDULE] 
	def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
	
		graph = collections.defaultdict(list) 

		for x, y in prerequisites: 
			graph[x].append(y)

		traced = set() 
		visited = set() 

		def dfs(k): 
			if k in traced:
				return False
		
			if k in visited:
				return True 

			traced.add(k) 

			for y_ in graph[k]: 
				if not dfs(y_):
					return False 

			traced.remove(k) 
			visited.add(k) 

			return True 

		for x_ in list(graph):
			if not dfs(x_):
				return False

		return True 					


# [LEETCODE #743 NETWORK DELAY TIME] 
	def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
		
		graph = collections.defaultdict(list) 
		for u, v, w in times: 
			graph[u].append((v, w)) 

		dist = collections.defaultdict(int) 
		queue = [(0, k)] 	# time, node

		while queue: 
			time, node = heapq.heappop(queue) 

			if node not in dist: 
				dist[node] = time 

				for v, w in graph[node]:
					alt = time + w 
					heapq.heappush(queue, (alt, v)) 

		if len(dist) == n: 
			return max(dist.values()) 

		return -1 

# [LEETCODE #787 CHEAPEST FLIGHTS WITHIN K STOPS] 
	def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
		
		graph = collections.defaultdict(list) 
		for x, y, z in flights: 
			graph[x].append((y, z)) 

		dist = collections.defaultdict(int) 
		queue = [(0, src, 0)] 	# price, node, count 

		while queue: 
			price, node, count = heapq.heappop(queue) 

			if node == dst and count - 1 <= k: 
				return price 

			if node not in dist or dist[node] > count: 
				dist[node] = count 

				for y, z in graph[node]: 
					alt = price + z 

					heapq.heappush(queue, (alt, y, count + 1))

		return -1 

# [LEETCODE #104 MAXIMUM DEPTH OF BINARY TREE]
	def maxDepth(self, root: Optional[TreeNode]) -> int:
		  
		if root is None:
			return 0 

		queue = collections.deque([root]) 
		depth = 0 

		while queue: 
			depth += 1 

			for _ in range(len(queue)): 
				cur_root = queue.popleft() 

				if cur_root.left:
					queue.append(cur_root.left) 
				if cur_root.right:
					queue.append(cur_root.right) 

		return depth 
	
	def maxDepthRecursive(self, root: Optional[TreeNode]) -> int: 
		def dfs(node):
			if node is None:
				return 0  

			left = dfs(node.left) 
			right = dfs(node.right) 

			return max(left, right) + 1
		
		return dfs(root)


# [LEETCODE #543 DIAMETER OF BINARY TREE] 
	path : int = 0
	def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
		def dfs(node):
			if node is None: 
				return 0 

			left = dfs(node.left) 
			right = dfs(node.right) 

			self.path = max(self.path, left + right) 

			return max(left, right) + 1 

		dfs(root) 

		return self.path  
			

# [LEETCODE #687 LONGEST UNIQUE VALUE PATH] 
	pathUnique: int = 0 
	def longestUnivaluePath(self, root: Optional[TreeNode]) -> int: 

		def dfs(node): 
			if node is None:
				return 0 

			left = dfs(node.left)
			right = dfs(node.right)

			if node.left and node.left.val == node.val: 
				left += 1 
			else: 
				left = 0 

			if node.right and node.right.val == node.val:
				right += 1 
			else:
				right = 0 

			self.pathUnique = max(self.pathUnique, left + right) 
		
			return max(left, right) 
		
		dfs(root)

		return self.pathUnique


# [LEETCODE #226 INVERT BINARY TREE] 
	def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:

		if root: 
			root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
	
			return root 

		return None 

		
	def invertTreeWhile(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
		queue = collections.deque([root]) 
       
		while queue: 
			node = queue.popleft() 
            
			if node: 
				node.left, node.right = node.right, node.left 
                
				queue.append(node.left)
				queue.append(node.right) 
                
		return root 


# [LEETCODE #617 MERGE TWO BINARY TREES] 
	def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]: 
		
		if root1 and root2: 
			node = TreeNode(root1.val + root2.val) 
		
			node.left = self.mergeTrees(root1.left, root2.left)
			node.right = self.mergeTrees(root1.right, root2.right) 

			return node 

		else:
			return root1 or root2 


# [LEETCODE #110 BALANCED BINARY TREE] 
	def isBalanced(self, root: Optional[TreeNode]) -> bool: 
		
		if root is None:
			return True 

		def dfs(node):
			if node is None:
				return True 
			
			left = dfs(node.left)
			right = dfs(node.right) 

			if left == -1 or right == -1 or abs(left - right) > 1:
				return -1 
		
			return max(left, right) + 1 

		return dfs(root) != -1 	
				

# [LEETCODE #310 MINIMUM HEIGHT TREES]
	def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:

		if n <= 1:
			return [0] 
		
		graph = collections.defaultdict(list) 
		for a, b in edges: 
			graph[a].append(b)
			graph[b].append(a) 

		leaves = [] 
		for i in range(n):
			if len(graph[i]) == 1: 
				leaves.append(i)

		while n > 2:
			n -= len(leaves) 
			
			new_leaves = []
			for leaf in leaves: 
				neighbor = graph[leaf].pop() 
				graph[neighbor].remove(leaf) 

				if len(graph[neighbor]) == 1: 
					new_leaves.append(neighbor)
					 	
			leaves = new_leaves 

		return leaves 
			

# [LEETCODE #108 CONVERT SORTED ARRAY TO BINARY SEARCH TREE] 
	def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
	
		if not nums:
			return None 

		mid = len(nums) // 2 
	
		node = TreeNode(nums[mid]) 

		node.left = self.sortedArrayToBST(nums[:mid]) 
		node.right = self.sortedArrayToBST(nums[mid + 1:]) 

		return node 

			
# [LEETCODE #1038 BINARY SEARCH TREE TO GREATER SUM TREE] 
	sum: int = 0 
	def bstToGst(self, root: TreeNode) -> TreeNode:

		def dfs(node):
			if node is None:
				return None 

			
			right = dfs(node.right) 

			self.sum += node.val 
			node.val = self.sum 

			left = dfs(node.left) 
	
			return node
		
		return dfs(root) 


# [LEETCODE #938 RANGE SUM OF BST] 
	rangeSum: int = 0 
	def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
	
		def dfs(node):
			if node:
				if node.val >= low and node.val <= high:
					self.rangeSum += node.val 

				left = dfs(node.left)
				right = dfs(node.right) 

		dfs(root)
		
		return self.rangeSum	


# [LEETCODE #783 MINIMUM DISTANCE BETWEEN BST NODES] 
	minDiff: int = sys.maxsize 
	prev: int = -sys.maxsize 
	def minDiffInBST(self, root: Optional[TreeNode]) -> int: 

		def dfs(node): 
			if node.left: 
				dfs(node.left) 

			self.minDiff = min(self.minDiff, node.val - self.prev) 
			self.prev = node.val 

			if node.right:
				dfs(node.right) 

		dfs(root) 

		return self.minDiff
