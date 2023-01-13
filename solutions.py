# Dependencies 
import re 
import collections 

# Soultions 
class Solution: 

# [LEETCODE #125. VALID PALINDROME]
# Given a string s, return true if it is a palindrome, or false otherwise.

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
# Write a function that reverses a string. The input string is given as an array of characters s.

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



# TEST CASE 
solution = Solution() 
print(solution.isPalindrome("A man, a plan, a canal: Panama")) 
print(solution.reverseString(['h', 'e', 'l', 'l', 'o']))  
print(solution.mostCommonWord('Bob hit a ball, the hit BALL flew far after it was hit.', 'hit'))
