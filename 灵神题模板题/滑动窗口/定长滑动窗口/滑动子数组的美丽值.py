# 定长窗口滑动从第二步开始就只需要判断首尾元素,
# 看到求一段，应该想到滑动窗口，前缀和，差分
# 定长窗口注意边界条件，0下标初始化，从1下标开始遍历，到len(arr)-k+1结束，下标为i时窗口的最后一位元素是arr[i+k-1]
# https://leetcode.cn/problems/sliding-subarray-beauty/  值域小可以参考计数排序
class Solution:
    def getSubarrayBeauty(self, nums, k, x):
        cnt = [0] * 101
        for num in nums[:k - 1]:  # 先往窗口内添加 k-1 个数
            cnt[num] += 1
        ans = [0] * (len(nums) - k + 1)
        for i, (in_, out) in enumerate(zip(nums[k - 1:], nums)):
            cnt[in_] += 1  # 进入窗口（保证窗口有恰好 k 个数）
            left = x
            for j in range(-50, 0):  # 暴力枚举负数范围 [-50,-1]
                left -= cnt[j]
                if left <= 0:  # 找到美丽值
                    ans[i] = j
                    break
            cnt[out] -= 1  # 离开窗口
        return ans
