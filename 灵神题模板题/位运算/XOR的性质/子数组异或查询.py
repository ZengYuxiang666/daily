# 要求一个区间的异或可以先建立一个异或前缀和子数组，求left,right之间的异或和时为pre_xor_sum[right]^pre_xor_sum[left-1]
# a⊕b⊕b=a，即自反性 a^0=a a^a=0
# https://leetcode.cn/problems/xor-queries-of-a-subarray
class Solution(object):
    def xorQueries(self, arr, queries):
        result = []
        pre_xor_sum = [arr[0]]
        for i in range(1,len(arr)):
            pre_xor_sum.append(pre_xor_sum[i-1] ^ arr[i])
        for left,right in queries:
            if left == 0:
                result.append(pre_xor_sum[right])
                continue
            result.append(pre_xor_sum[right]^pre_xor_sum[left-1])
        return result