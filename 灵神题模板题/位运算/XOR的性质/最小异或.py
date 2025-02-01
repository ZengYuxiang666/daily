# 分为两种情况：c1>c2或者c2>c1，当c2小于c1时，要使x^num1的值最小，即是将num1前c2位1置为0，而num1 - 1是将最低的1置为0
# 反向来说就是将num1的后len(num1)-c2位值为0，得到的就是x的值
# 当c2大于c1时，可以用c1位1将num1全置为0，然后多的c2-c1个1将他放到最后面，即是将num1的后c2-c1个0置为1

# https://leetcode.cn/problems/minimize-xor

# 对位运算要有集合的思维，https://leetcode.cn/circle/discuss/CaOJ45/
class Solution:
    def minimizeXor(self, num1: int, num2: int) -> int:
        c1 = num1.bit_count()
        c2 = num2.bit_count()
        while c2 < c1:
            num1 &= num1 - 1  # 将num1中最低的 1 变成 0
            c2 += 1
        while c2 > c1:
            num1 |= num1 + 1  # 将num1中最低的 0 变成 1
            c2 -= 1
        return num1