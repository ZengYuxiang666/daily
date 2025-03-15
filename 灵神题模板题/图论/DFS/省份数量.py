# 当题目用n*n的数组表示边时，先建立一个数组表示每个节点是不是被访问过，每访问一个节点时遍历整行
# https://leetcode.cn/problems/number-of-provinces
class Solution:
    def findCircleNum(self, isConnected) -> int:
        n = len(isConnected)  # 城市的数量
        visited = [False] * n  # 记录每个城市是否被访问过

        def dfs(city):
            for neighbor in range(n):
                if isConnected[city][neighbor] == 1 and not visited[neighbor]:
                    visited[neighbor] = True
                    dfs(neighbor)

        province_count = 0  # 记录省份的数量
        for i in range(n):
            if not visited[i]:  # 如果城市 i 没有被访问过
                dfs(i)  # 从城市 i 开始深度优先遍历
                province_count += 1  # 找到一个新的连通分量，省份数量加 1

        return province_count