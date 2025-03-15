# 找他的最小高度树即是找树的中心，从每个叶节点慢慢向中心逼近，每次将叶节点抹去，当一个节点的度为一时又为一个新的叶节点
# 用while n > 2:的原因是当树的直径(树的最大长度)为偶数时，中心节点有2个，为奇数时的中心节点有1个
# https://leetcode.cn/problems/minimum-height-trees
from collections import defaultdict, deque
from typing import List
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        # 如果节点数量为 1，直接返回 [0]
        if n == 1:
            return [0]

        # 构建邻接表
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # 初始化叶子节点队列
        leaves = deque([i for i in range(n) if len(adj[i]) == 1])
        # 去除叶子节点，直到剩下中心节点
        while n > 2:
            size = len(leaves)
            n -= size
            for _ in range(size):
                leaf = leaves.popleft()
                for neighbor in adj[leaf]:
                    adj[neighbor].remove(leaf)
                    if len(adj[neighbor]) == 1:
                        leaves.append(neighbor)

        # 剩下的节点就是最小高度树的根节点
        return list(leaves)