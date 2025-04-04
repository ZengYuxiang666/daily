# 记忆化搜索，选s[i]则j更新，不选则不更新，j到头即匹配成功一个，返回1
# https://leetcode.cn/problems/distinct-subsequences
from functools import cache
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        @cache
        def dfs(i,j):
            if j<0:  # j到头代表选完了
                return 1
            if i < 0:
                return 0
            if s[i] == t[j]:  # 假如匹配s可以选或者不选
                return dfs(i-1,j-1)+dfs(i-1,j)  # 第一个是s选，第二个是s不选
            else: # 假如不匹配，i一定不能选
                return dfs(i-1,j)
        return dfs(len(s)-1,len(t)-1)