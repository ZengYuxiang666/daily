# https://leetcode.cn/problems/word-break
#
from functools import cache
from typing import List
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        @cache
        def dfs(i):
            if i < 0:
                return True
            for word in wordDict:
                len_word = len(word)
                if s[i-len_word+1:i+1]==word:
                    if(dfs(i-len_word)):
                        return True
            return False

        return dfs(len(s)-1)
p = Solution()
print(p.wordBreak(s = "catsandog", wordDict =["cats","dog","sand","and","cat"]))