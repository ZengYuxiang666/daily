# 求第k小的话，check：找出比num小的数，当total>=k时返回true，否则返回False，if(check(mid))==true:right = mid,return left
# 求第k大的话，check：找出比num大的数，当total>=k时返回true，否则返回False,if(check(mid))==ture:left = mid+1,return left - 1
# https://leetcode.cn/problems/find-k-th-smallest-pair-distance
class Solution(object):
    def smallestDistancePair(self, nums, k):
        nums.sort()

        def check(num):
            total = 0
            right = 0
            for left in range(len(nums)):
                while right < len(nums) and nums[right] - nums[left] <= num:
                    right += 1
                total += right - left - 1  # right-left-1 是满足条件的数对数
            return total >= k

        left = 0
        right = nums[-1] - nums[0]
        while left < right:
            mid = (left + right) // 2
            if check(mid):
                right = mid
            else:
                left = mid + 1
        return left
