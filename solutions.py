# Dependencies 
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

		for idx, num  in enumerate(nums): 
			complement = target - num
			
			if complement in nums: 
				return [idx, nums[idx + 1:].index(complement) + (idx + 1)]





# TEST CASE 
solution = Solution() 

print(solution.isPalindrome("A man, a plan, a canal: Panama")) 
print(solution.reverseString(['h', 'e', 'l', 'l', 'o']))  
print(solution.mostCommonWord('Bob hit a ball, the hit BALL flew far after it was hit.', 'hit'))
print(solution.longestPalindrome('asdlkaasndssssslkkknaawww'))
