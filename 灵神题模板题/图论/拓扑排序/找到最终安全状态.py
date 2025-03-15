# 反向拓扑，首先所有入度为0的节点为安全节点，加入queue中，然后进行bfs遍历后所以入度变为0的节点也为安全节点
# https://leetcode.cn/problems/find-eventual-safe-states
from collections import deque
from typing import List
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        rg = [[] for _ in graph]
        for x, ys in enumerate(graph):
            for y in ys:
                rg[y].append(x)
        in_deg = [len(ys) for ys in graph]

        q = deque([i for i, d in enumerate(in_deg) if d == 0])
        while q:
            for x in rg[q.popleft()]:
                in_deg[x] -= 1
                if in_deg[x] == 0:
                    q.append(x)

        return [i for i, d in enumerate(in_deg) if d == 0]
