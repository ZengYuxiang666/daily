# 打家劫舍就是当前这一个不要，前一个就不能要，当前这个要的话，那就要那前面第2个,因为就算前面那个最大化是不要的，i-1也是等于i-2。
# https://leetcode.cn/problems/delete-and-earn
from collections import Counter
class Solution(object):
    def deleteAndEarn(self, nums):
        if not nums:
            return 0

        # 统计每个数的总点数
        count = Counter(nums)
        max_num = max(nums)  # 找到 nums 中的最大值
        points = [0] * (max_num + 1)  # 创建一个数组存储每个数的总点数

        for num in count:
            points[num] = num * count[num]  # 计算每个数的总得分

        # 处理 DP 转移
        dp = [0] * (max_num + 1)
        dp[1] = points[1]  # 只有 1 时，最多能拿 points[1]

        for i in range(2, max_num + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + points[i])  # 经典打家劫舍公式

        return dp[max_num]  # 返回最大点数
