# 相当于最长公共子序列，但是a数组必须选完，a没选完的返回负无穷
# 这个必选看原题，题目上很好的体现了最长公共子序列的顺序不能变的特点
# https://leetcode.cn/problems/maximum-multiplication-score
from functools import cache
from typing import List
class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        @cache
        def dfs(i,j):
            if i<0 :
                return 0
            if j <0:
                return float('-inf')
            return max(dfs(i-1,j-1)+a[i]*b[j],dfs(i,j-1))
        return dfs(len(a)-1,len(b)-1)