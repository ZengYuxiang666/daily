"""
对于任意的区间 [i,j)，其异或值可以通过 prefix_xor[j] ^ prefix_xor[i] 计算出来。
对于任意的区间 [j,k]，其异或值可以通过 prefix_xor[k+1] ^ prefix_xor[j] 计算出来。
要两个区间的异或值相等，则prefix_xor[j]=prefix_xor[k+1]
"""
# https://leetcode.cn/problems/count-triplets-that-can-form-two-arrays-of-equal-xor
class Solution:
    def countTriplets(self, arr: List[int]) -> int:
        n = len(arr)
        s = [0]
        for val in arr:
            s.append(s[-1] ^ val)

        ans = 0
        for i in range(n):
            for k in range(i + 1, n):
                if s[i] == s[k + 1]:
                    ans += k - i

        return ans
