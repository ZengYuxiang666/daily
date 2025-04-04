# 和销售利润最大化差不多，不同的是要初始化路费，和idx = bisect.bisect_right(ends, start) - 1 因为车站可以从一站停然后再从这站开始
import bisect
from typing import List
from functools import cache

class Solution:
    def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
        # 将路费初始化
        for i in range(len(rides)):
            rides[i][2] = rides[i][2] + rides[i][1] - rides[i][0]
        rides.sort(key=lambda x:x[1])
        ends = [ride[1] for ride in rides]
        @cache
        def dfs(i):
            if i < 0:
                return 0

            # 当前报价
            start, end, gold = rides[i]

            # 使用二分查找找到前一个不冲突报价
            idx = bisect.bisect_right(ends, start) - 1

            # 选当前报价
            take = gold+(dfs(idx) if idx >= 0 else 0)

            # 不选当前报价
            skip = dfs(i - 1)

            # 返回最大收益
            return max(take, skip)
        # 从最后一个报价开始计算
        return dfs(len(rides) - 1)
p = Solution()
print(p.maxTaxiEarnings(n = 20, rides = [[1,6,1],[3,10,2],[10,12,3],[11,12,2],[12,15,2],[13,18,1]]))