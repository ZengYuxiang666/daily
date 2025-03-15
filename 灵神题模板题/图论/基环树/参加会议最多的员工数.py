# 对于长度大于 2 的基环，取基环长度的最大值；
# 对于长度等于 2 的基环，可以从基环上的点出发，在反图上找到最大的树枝节点深度。 具体看零神题解
# https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting
# 我感觉基环树就是进行一个拓扑排序进行剪枝，使其只剩下一个环
from collections import deque
from typing import List
class Solution:
    def maximumInvitations(self, favorite: List[int]) -> int:
        n = len(favorite)
        deg = [0] * n
        for f in favorite:
            deg[f] += 1  # 统计基环树每个节点的入度

        rg = [[] for _ in range(n)]  # 反图
        q = deque(i for i, d in enumerate(deg) if d == 0)
        while q:  # 拓扑排序，剪掉图上所有树枝
            x = q.popleft()
            y = favorite[x]  # x 只有一条出边
            rg[y].append(x)
            deg[y] -= 1
            if deg[y] == 0:
                q.append(y)

        # 通过反图 rg 寻找树枝上最深的链
        def rdfs(x: int) -> int:
            max_depth = 1
            for son in rg[x]:
                max_depth = max(max_depth, rdfs(son) + 1)
            return max_depth

        max_ring_size = sum_chain_size = 0
        for i, d in enumerate(deg):
            if d == 0: continue  # 因为剪掉了树枝，入度为0的节点是为数值，不要遍历

            # 遍历基环上的点
            deg[i] = 0  # 将基环上的点的入度标记为 0，避免重复访问
            ring_size = 1  # 基环长度
            x = favorite[i]
            while x != i:
                deg[x] = 0  # 将基环上的点的入度标记为 0，避免重复访问
                ring_size += 1
                x = favorite[x]

            if ring_size == 2:  # 基环长度为 2
                # 因为只有两个节点，所以树的直径为两个节点的反向最长路径之和
                sum_chain_size += rdfs(i) + rdfs(favorite[i])  # 累加两条最长链的长度
            else:
                max_ring_size = max(max_ring_size, ring_size)  # 取所有基环长度的最大值
        return max(max_ring_size, sum_chain_size)