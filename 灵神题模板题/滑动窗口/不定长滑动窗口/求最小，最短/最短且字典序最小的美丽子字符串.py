# 求最短和最长的思路大体相同，用for循环移动right指针，当满足条件时更新最小长度，然后移动left指针
# https://leetcode.cn/problems/shortest-and-lexicographically-smallest-beautiful-string
class Solution(object):
    def shortestBeautifulSubstring(self, s, k):
        left = 0
        lst = []
        total = 0
        min_length = float('inf')

        for right in range(len(s)):
            if s[right] == '1':
                total += 1

            while total >= k:
                current_length = right - left + 1
                if current_length < min_length:
                    min_length = current_length
                    lst = [s[left:right + 1]]
                elif current_length == min_length:
                    lst.append(s[left:right + 1])

                if s[left] == '1':
                    total -= 1
                left += 1

        if not lst:
            return ""

        lst.sort()
        return lst[0]