# 贡献法即求每个元素的贡献
'''
核心思想：
对于每个元素 arr[i]，计算它作为子数组最小值的贡献。具体来说，arr[i] 在某些子数组中会是最小值，我们需要计算这些子数组的个数，并将它们的贡献加到总和中。
计算每个元素作为最小值的范围，可以使用单调栈来帮助我们找到每个元素作为最小值的左右边界。

具体步骤：
单调栈的作用：我们利用单调栈来记录数组元素的 下一个小于元素的索引 (next smaller) 和 前一个小于元素的索引 (previous smaller)。
prev_smaller[i] 表示 arr[i] 之前的一个元素，且所有索引在此之前的元素都大于 arr[i]。
next_smaller[i] 表示 arr[i] 之后的一个元素，且所有索引在此之后的元素都大于 arr[i]。
贡献计算：对于每个 arr[i]，它作为子数组最小值的子数组数量是 (i - prev_smaller[i]) * (next_smaller[i] - i)，即：
它可以作为子数组最小值的子数组起点数是 (i - prev_smaller[i])。
它可以作为子数组最小值的子数组终点数是 (next_smaller[i] - i)。
总和计算：每个 arr[i] 的贡献是它本身的值乘以上述计算得到的子数组数量。
'''

# 用单调栈快速地求比该元素大的元素的下标
# https://leetcode.cn/problems/sum-of-subarray-minimums
class Solution:
    def sumSubarrayMins(self, arr):
        MOD = 10 ** 9 + 7
        n = len(arr)

        # prev_smaller[i] 存储的是 arr[i] 左边第一个比 arr[i] 小的元素的索引
        prev_smaller = [-1] * n
        # next_smaller[i] 存储的是 arr[i] 右边第一个比 arr[i] 小的元素的索引
        next_smaller = [n] * n

        # 单调栈，用来计算 prev_smaller 和 next_smaller
        stack = []

        # 计算 prev_smaller
        for i in range(n):
            while stack and arr[stack[-1]] >= arr[i]:
                stack.pop()
            if stack:
                prev_smaller[i] = stack[-1]
            stack.append(i)

        # 清空栈，计算 next_smaller
        stack.clear()
        for i in range(n - 1, -1, -1):
            while stack and arr[stack[-1]] > arr[i]:
                stack.pop()
            if stack:
                next_smaller[i] = stack[-1]
            stack.append(i)

        # 计算结果
        result = 0
        for i in range(n):
            # 每个 arr[i] 的贡献是它的值乘以它作为最小值的子数组数量
            result += arr[i] * (i - prev_smaller[i]) * (next_smaller[i] - i)
            result %= MOD

        return result