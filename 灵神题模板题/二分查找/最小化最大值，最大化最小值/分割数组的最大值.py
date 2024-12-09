# 最小化最大值：不断查找num，check函数的返回值由num倒推题目条件成不成立，if check(num)==True,right=mid;
# 最小化最大值与上面差不多
# https://leetcode.cn/problems/split-array-largest-sum
class Solution(object):
    def splitArray(self, nums, k):
        def check(num):
            total = 1
            x = 0
            for i in nums:
                if i > num:
                    return False
                if x + i > num:
                    x = i
                    total += 1
                else:
                    x += i
            return total <= k
        left = max(nums)
        right = sum(nums)+1
        while left < right:
            mid = (left+right)//2
            if check(mid):
                right = mid
            else:
                left = mid + 1
        return left