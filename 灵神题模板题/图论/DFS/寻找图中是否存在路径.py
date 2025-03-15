# 当题目是给出哪两节点相邻时，应该构建领接表如何遍历
# https://leetcode.cn/problems/find-if-path-exists-in-graph
class Solution:
    def validPath(self, n: int, edges, source: int, destination: int) -> bool:
        # 创建邻接表
        graph = {i: [] for i in range(n)}
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # DFS
        def dfs(current, destination, visited):
            if current == destination:
                return True
            visited.add(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    if dfs(neighbor, destination, visited):
                        return True
            return False

        visited = set()
        return dfs(source, destination, visited)