# https://leetcode.cn/problems/maximum-candies-allocated-to-k-children
class Solution(object):
    def maximumCandies(self, candies, k):
        def check(num):
            total = 0
            for i in candies:
                total += i//num
            return total >= k
        left = 1
        right = max(candies)+1
        while left < right:
            mid = (left+right)//2
            if check(mid):
                left = mid + 1
            else:
                right = mid
        return left -1