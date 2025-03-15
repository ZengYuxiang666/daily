# 适用于网格图
# https://leetcode.cn/problems/find-minimum-time-to-reach-last-room-i
# 在我看来这种算法的思路是，建立一个表，表示到各个节点的最短时间，然后维护一个最小堆，每次弹出的都是已经确定的最短时间的节点
# 然后在这个基础上往他能走的地方遍历，更新其他的节点
import heapq
class Solution(object):
    def minTimeToReach(self, moveTime):
        n, m = len(moveTime), len(moveTime[0])

        # 创建一个二维数组来存储从 (0, 0) 到每个房间的最短时间
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0  # 从起点开始，初始时间是0

        # 优先队列，用于 Dijkstra 算法
        pq = [(0, 0, 0)]  # (当前时间, x, y)

        # 定义四个方向：上下左右
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while True:
            current_time, x, y = heapq.heappop(pq)
            if x == n - 1 and y == m - 1:
                return current_time

            # 如果当前房间已经到达了最优解，则跳过
            if current_time > dist[x][y]:
                continue

            # 遍历四个方向
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # 判断是否越界
                if 0 <= nx < n and 0 <= ny < m:
                    # 计算进入相邻房间的时间
                    # 当前时间 + 1秒移动时间，和 moveTime[nx][ny] 房间的时间的最大值
                    new_time = max(current_time + 1, moveTime[nx][ny] + 1)

                    # 如果通过这个房间的时间更短，则更新并加入优先队列
                    if new_time < dist[nx][ny]:
                        dist[nx][ny] = new_time
                        heapq.heappush(pq, (new_time, nx, ny))
