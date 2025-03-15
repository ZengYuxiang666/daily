# 不含相邻0可以转换成其反码不含相邻1,反码不含相邻1可以转换成与其往左缩进一位的数等于0  x & (x >> 1) 等于 0
# https://leetcode.cn/problems/generate-binary-strings-without-adjacent-zeros
class Solution:
    def validStrings(self, n):
        ans = []
        mask = (1 << n) - 1
        for x in range(1 << n):
            if (x >> 1) & x == 0:
                # 0{n}b 表示长为 n 的有前导零的二进制
                ans.append(f"{x ^ mask:0{n}b}")
        return ans
