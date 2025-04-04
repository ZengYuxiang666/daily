# 假如s[j:i+1]在字典中，则操作次数不用变，假如不在，则加上子数组的长度，对于每个i，从0开始找，直到找到最优解
# https://leetcode.cn/problems/extra-characters-in-a-string
from functools import cache
from typing import List
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        set_dictionary = set(dictionary)
        @cache
        def dfs(i):
            res = float('inf')
            if i == -1:
                return 0
            else:
                for j in range(i+1):
                    if s[j:i+1] in set_dictionary:
                        res = min(res,dfs(j-1))
                    else:
                        res = min(res,dfs(j-1)+i-j+1)
            return res
        return dfs(len(s)-1)
p = Solution()
print(p.minExtraChar(s = "sayhelloworld", dictionary = ["hello","world"]))
