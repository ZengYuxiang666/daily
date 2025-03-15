# 这才是最正宗的bfs，用了一个数组dist记录了起始节点到其他节点的最短距离
# 
from collections import deque, defaultdict
class Solution:
    def findShortestCycle(self, n: int, edges) -> int:
        # 第一步：构建图（邻接表）
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # 第二步：BFS 函数，用于从给定节点出发查找最短环
        def bfs(start):
            dist = [-1] * n  # 记录从起始节点到其他节点的距离
            parent = [-1] * n  # 记录每个节点的父节点，避免回到父节点
            dist[start] = 0
            queue = deque([start])

            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    # 如果邻居节点还未被访问过
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    # 如果邻居节点已被访问过，且不是当前节点的父节点，则发现了一个环
                    elif parent[node] != neighbor:
                        # 计算环的长度
                        return dist[node] + dist[neighbor] + 1

            return float('inf')  # 如果没有找到环，返回无穷大

        # 第三步：遍历所有节点，找到最短的环
        shortest_cycle = float('inf')

        for i in range(n):
            shortest_cycle = min(shortest_cycle, bfs(i))

        return -1 if shortest_cycle == float('inf') else shortest_cycle