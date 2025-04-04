# 用的是dfs加上记忆化搜索，本质是选或者不选
# https://leetcode.cn/problems/delete-operation-for-two-strings
from functools import cache
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        @cache
        def dfs(i, j):
            if i < 0:
                return j + 1  # 删除 word2 剩下的字符
            if j < 0:
                return i + 1  # 删除 word1 剩下的字符

            if word1[i] == word2[j]:
                return dfs(i - 1, j - 1)

            # 删除 word1[i] 或 word2[j]，步数+1
            return min(dfs(i - 1, j) + 1, dfs(i, j - 1) + 1)

        return dfs(len(word1) - 1, len(word2) - 1)
