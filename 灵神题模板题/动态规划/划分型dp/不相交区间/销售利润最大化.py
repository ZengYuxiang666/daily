# 先给结束时间排序，然后从最后一个方案开始就是选或者不选的问题，选的话要用二分搜索下一个不相交的选择，
# 要特别注意bisect.bisect_right的使用
# https://leetcode.cn/problems/maximize-the-profit-as-the-salesman
import bisect
from functools import cache
from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # 按结束时间排序
        offers.sort(key=lambda x: x[1])
        print(offers)
        # 提取结束时间，便于二分查找
        ends = [offer[1] for offer in offers]

        @cache
        def dfs(i):
            if i < 0:
                return 0

            # 当前报价
            start, end, gold = offers[i]

            # 使用二分查找找到前一个不冲突报价
            idx = bisect.bisect_right(ends, start - 1) - 1

            # 选当前报价
            take = gold + (dfs(idx) if idx >= 0 else 0)

            # 不选当前报价
            skip = dfs(i - 1)

            # 返回最大收益
            return max(take, skip)

        # 从最后一个报价开始计算
        return dfs(len(offers) - 1)

# ✅ 测试
p = Solution()
print(p.maximizeTheProfit(4, [[1, 3, 10], [1, 3, 3], [0, 0, 1], [0, 0, 7]]))   # 输出：11
print(p.maximizeTheProfit(5, [[0, 0, 1], [0, 2, 2], [1, 3, 2]]))               # 输出：3
print(p.maximizeTheProfit(5, [[0, 1, 3], [2, 3, 4], [3, 4, 5]]))