from functools import cache
# 得到的子问题有两种解决思路 1；选或者不选，需要知道上一个选的数字  2：枚举选哪个，比较当前选的数字和下一个要选的数字  很明显第二种方法只要一个参数
# https://leetcode.cn/problems/longest-increasing-subsequence
class Solution:
    def lengthOfLIS(self, nums):
        @cache
        def dfs(i):  # 计算以num[i]结尾的最长递增子序列的长度
            res = 0
            for j in range(i):
                if nums[j] < nums[i]:
                    res = max(res,dfs(j))  # 因为是枚举，所以每个符合条件的都选上
            res += 1  # 代表选上了nums[i]
            return res
        res = 0
        for i in range(len(nums)):
            res = max(dfs(i),res)  # 枚举，枚举以每个数字结尾的最长递增子序列长度
        return res



