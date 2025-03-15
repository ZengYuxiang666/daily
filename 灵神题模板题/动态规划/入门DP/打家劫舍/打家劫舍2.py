# 下次有环形的打家劫舍就这样做，第一个打劫了最后一个就不能打劫了，最后一个打劫了第一个就不能打劫了
# https://leetcode.cn/problems/house-robber-ii
class Solution(object):
    def rob(self, nums):
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]

        def rob_range(start, end):
            n = end - start + 1
            if n == 1:
                return nums[start]

            dp = [0] * n
            dp[0] = nums[start]
            dp[1] = max(nums[start], nums[start + 1])

            for i in range(2, n):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[start + i])

            return dp[-1]

        return max(rob_range(0, len(nums) - 2), rob_range(1, len(nums) - 1))  # 关键所在
