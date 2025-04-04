# 插入和删除都是同一个状态转移方程
# https://leetcode.cn/problems/edit-distance
from functools import cache
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        @cache
        def dfs(i,j):
            if(i<0):
                return j+1
            elif(j<0):
                return i+1
            if(word1[i]==word2[j]):
                return dfs(i-1,j-1)
            else:
                return min(dfs(i-1,j)+1,dfs(i,j-1)+1,dfs(i-1,j-1)+1)
        return dfs(len(word1)-1,len(word2)-1)