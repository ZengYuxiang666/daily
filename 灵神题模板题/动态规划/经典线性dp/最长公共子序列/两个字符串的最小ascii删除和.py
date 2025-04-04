# https://leetcode.cn/problems/minimum-ascii-delete-sum-for-two-strings
from functools import cache
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        @cache
        def dfs(i,j):
            if i < 0:
                return sum([ord(i) for i in s2[:j+1]])
            if j < 0:
                return sum([ord(i) for i in s1[:i+1]])
            if  s1[i] == s2[j]:
                return dfs(i-1,j-1)
            return min(dfs(i-1,j)+ord(s1[i]),dfs(i,j-1)+ord(s2[j]))
        return dfs(len(s1)-1,len(s2)-1)