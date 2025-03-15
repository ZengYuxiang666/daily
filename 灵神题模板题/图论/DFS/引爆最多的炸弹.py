# 最关键的就是根据题目做出邻接表，我决定以后做任何图论题目都构建邻接表来做
# https://leetcode.cn/problems/detonate-the-maximum-bombs
import math
from typing import List
class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        # 判断炸弹 i 是否能引爆炸弹 j
        def canDetonate(bomb1, bomb2):
            x1, y1, r1 = bomb1
            x2, y2, r2 = bomb2
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            return distance <= r1  # 如果炸弹2在炸弹1的爆炸范围内

        n = len(bombs)
        graph = {i: [] for i in range(n)}

        # 构建邻接表
        for i in range(n):
            for j in range(n):
                if i != j and canDetonate(bombs[i], bombs[j]):
                    graph[i].append(j)

        # 深度优先搜索（DFS）来遍历并计算最大引爆数
        def dfs(node, visited):
            visited[node] = True
            count = 1  # 计入当前炸弹
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    count += dfs(neighbor, visited)
            return count

        # 对每个炸弹尝试引爆，记录最大值
        max_bombs = 0
        for i in range(n):
            visited = [False] * n
            max_bombs = max(max_bombs, dfs(i, visited))

        return max_bombs