# check函数直接查找的num不是答案，而是一个和答案有关的值（间接值）
# https://leetcode.cn/problems/maximum-points-inside-the-square
# 该题用check函数查找半径，半径最大且check函数成立时包含的点肯定最多，所以check查找的不是答案，而是与他有关的值
class Solution(object):
    def maxPointsInsideSquare(self, points, s):
        # 计算每个点到原点的距离
        distances = [max(abs(point[0]), abs(point[1])) for point in points]

        def check(num):
            used_labels = set()
            for i in range(len(distances)):
                if distances[i] <= num:
                    if s[i] in used_labels:
                        return -1
                    used_labels.add(s[i])
            return len(used_labels)

        left, right = 0, max(distances)+1
        max_points = 0

        while left < right:
            mid = (left + right) // 2
            if check(mid) >= 0:
                max_points = max(max_points, check(mid))
                left = mid + 1
            else:
                right = mid

        return max_points