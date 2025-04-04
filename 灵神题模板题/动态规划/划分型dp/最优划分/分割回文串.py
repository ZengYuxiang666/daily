from functools import cache
# https://leetcode.cn/problems/palindrome-partitioning-ii
class Solution:
    def minCut(self, s: str) -> int:
        # 返回 s[l:r+1] 是否为回文串
        @cache  # 缓存装饰器，避免重复计算 is_palindrome（一行代码实现记忆化）
        def is_palindrome(l: int, r: int) -> bool:
            if l >= r:
                return True
            return s[l] == s[r] and is_palindrome(l + 1, r - 1)

        @cache  # 缓存装饰器，避免重复计算 dfs（一行代码实现记忆化）
        def dfs(r: int) -> int:
            if is_palindrome(0, r):  # 已是回文串，无需分割
                return 0
            res = float('inf')
            for l in range(1, r + 1):  # 枚举分割位置
                if is_palindrome(l, r):
                    res = min(res, dfs(l - 1) + 1)  # 在 l-1 和 l 之间切一刀
            return res

        return dfs(len(s) - 1)
