# 求最小时间：check函数不断查找num，num越大越有可能成立，成立时移动右指针求成立时的最小值，记住！不管求最大或最小，left指针始终等于mid+1
# 因为mid=(left+right)//2,当left指针和right指针相邻时最后退出时的left指针指向right的位置，且不会进入死循环(例：left=2，right=3时)
# while(left<right)始终不会退出
# 当求最小时:return left,例：[false,false,true,true,true],当最后一个循环时left=1,right=2,因为mid=(left+right)//2,所以最后
# 退出时left = 2,reyurn left 符合求最小值   而求最大值时[ture,ture,false,false,false],退出循环时left=2，lst[left]==False,
# 所以求最大时 return left-1
# https://leetcode.cn/problems/minimum-time-to-repair-cars
import math
class Solution(object):
    def repairCars(self, ranks, cars):
        def check(num):
            total = 0
            for i in ranks:
                total += int(math.sqrt(num//i))
            return total >= cars
        left = 1
        right = 2**64-1
        while left < right:
            mid = (left+right)//2
            if check(mid):
                right = mid
            else:
                left = mid + 1
        return left