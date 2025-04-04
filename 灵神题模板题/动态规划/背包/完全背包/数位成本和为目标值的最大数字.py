# https://leetcode.cn/problems/form-largest-integer-with-digits-that-add-up-to-target
class Solution(object):
    def largestNumber(self, cost, target):
        # dp[i]存储目标为i时能够拼出的最大数字
        dp = ["0"] * (target + 1)
        dp[0] = ""  # 凑出0的代价是空字符串

        # 从大到小遍历数字（保证字典序最大）
        for digit in range(9, 0, -1):   # 遍历数字 9 -> 1
            c = cost[digit - 1]          # 当前数字的代价
            for v in range(c, target + 1):
                if dp[v - c] != "0" or v == c:
                    # 拼接当前数字到字符串前面
                    new_number = dp[v - c] + str(digit)
                    # 按字典序和长度取更大的字符串
                    if len(new_number) > len(dp[v]) or (len(new_number) == len(dp[v]) and new_number > dp[v]):
                        dp[v] = new_number

        return dp[target] if dp[target] != "0" else "0"