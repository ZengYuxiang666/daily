# https://leetcode.cn/problems/best-time-to-buy-and-sell-stock
from typing import List
from functools import cache
# dfs(i,0)第i天结束未持有股票的最大利润 dfs(i,1)第i天结束持有股票的最大利润
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        @cache
        def dfs(i,state):
            if i < 0:
                return float('-inf') if state else 0
            # 假如前一天也没持有 或者 前天持有今天卖了的最大值
            if state == 0:
                return max(dfs(i-1,0),dfs(i-1,1)+prices[i])
            else:
                # 前一天持有 或者 前一天未持有今天买了,因为只能买一次，所以是-prices[i]
                return max(dfs(i-1,1),-prices[i])
        return dfs(len(prices)-1,0)
p = Solution()
print(p.maxProfit([7,6,4,3,1]))