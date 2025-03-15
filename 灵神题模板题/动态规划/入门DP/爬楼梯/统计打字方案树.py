# 爬楼梯的题目就是这一阶梯的答案来源于前一阶答案
# https://leetcode.cn/problems/count-number-of-texts
class Solution(object):
    def countTexts(self, pressedKeys):
        MOD = 10 ** 9 + 7
        n = len(pressedKeys)
        dp = [0] * (n + 1)
        dp[0] = 1  # 空字符串只有一种可能

        for i in range(1, n + 1):
            dp[i] = dp[i - 1]  # 只考虑当前字符单独作为一个字母

            # 处理重复的按键字符
            if i > 1 and pressedKeys[i - 1] == pressedKeys[i - 2]:
                dp[i] = (dp[i] + dp[i - 2]) % MOD

            if i > 2 and pressedKeys[i - 1] == pressedKeys[i - 2] == pressedKeys[i - 3]:
                dp[i] = (dp[i] + dp[i - 3]) % MOD

            if i > 3 and pressedKeys[i - 1] == pressedKeys[i - 2] == pressedKeys[i - 3] == pressedKeys[i - 4] and \
                    pressedKeys[i - 1] in "79":
                dp[i] = (dp[i] + dp[i - 4]) % MOD

        return dp[n]