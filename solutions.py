# Dependencies 
import sys 
import re 
import collections 

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


# [LEETCODE #1. Two Sum] 
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





# TEST CASE 
solution = Solution() 

print(solution.isPalindrome("A man, a plan, a canal: Panama")) 
print(solution.reverseString(['h', 'e', 'l', 'l', 'o']))  
print(solution.mostCommonWord('Bob hit a ball, the hit BALL flew far after it was hit.', 'hit'))
print(solution.longestPalindrome('asdlkaasndssssslkkknaawww'))
print(solution.twoSum(nums = [3, 2, 4], target = 6))
print(solution.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
print(solution.threeSum(nums = [-1, 1, 3, -2, 2, -3]))
print(solution.productExceptSelf([1, 2, 3, 4, 5]))
print(solution.maxProfit([7, 1, 5, 3, 6, 4]))
