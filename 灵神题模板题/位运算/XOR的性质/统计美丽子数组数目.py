from itertools import accumulate
from collections import Counter
from operator import xor
# accumulate：accumulate 函数返回一个累积和（或累积操作）的迭代器。它可以用来计算数组的前缀和，也可以用来进行其他的累积操作（如异或）。
# Counter：是一个用于统计元素出现次数的字典子类。它会将每个元素作为键，并记录它们出现的次数。
# xor：是按位异或操作符的函数形式。xor(a, b) 与 a ^ b 的功能相同。

# 子数组为0既是前缀数组中i，j相同，以字典序记录更方便
class Solution:
    def beautifulSubarrays(self, nums):
        s = list(accumulate(nums, xor, initial=0))
        print(s)
        ans, cnt = 0, Counter()
        for x in s:
            # 先计入答案再统计个数，如果反过来的话，就相当于把空子数组也计入答案了
            ans += cnt[x]
            cnt[x] += 1
        return ans
p = Solution()
print(p.beautifulSubarrays([4,3,1,2,4]))