# BFS基础，我最关键的是第19行代码for _ in range(len(queue)):，这样就可以区分出是哪一步了
# https://leetcode.cn/problems/shortest-distance-after-road-addition-queries-i
from collections import deque, defaultdict
class Solution(object):
    def shortestDistanceAfterQueries(self, n, queries):
        graph = defaultdict(list)
        for i in range(n - 1):
            graph[i].append(i + 1)

        # 存储查询的结果
        result = []

        def bfs():
            queue = deque([0])  # bfs从0节点开始遍历
            valid = [0]*n  # 表示是否访问过
            step = 0  # 表示步数

            while queue:
                for _ in range(len(queue)):
                    city = queue.popleft()
                    valid[city] = 1
                    if city == n-1:
                        return step
                    for neighbor in graph[city]:
                        if valid[neighbor] == 0: # 没有访问过
                            queue.append((neighbor))
                step += 1  # 步数+1
        # 处理每一个查询
        for u, v in queries:
            # 将新道路添加到图中
            graph[u].append(v)

            # 使用 BFS 计算最短路径
            result.append(bfs())

        return result