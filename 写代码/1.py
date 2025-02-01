from collections import defaultdict


class Solution(object):
    def minMalwareSpread(self, graph, initial):
        edges = defaultdict(list)
        n = len(graph)

        # 构建图的邻接表
        for i in range(n):
            for j in range(n):
                if graph[i][j] == 1:
                    edges[i].append(j)
                    edges[j].append(i)

        initial_set = set(initial)
        valid = [0] * n  # 记录节点是否已经被访问过
        result = []

        def dfs(node, total):
            valid[node] = 1
            total[0] += 1  # 递归时增加节点的传播数量
            for i in edges[node]:
                if i in initial_set:
                    total[1] = True  # 标记是否包含初始感染节点
                elif not valid[i]:
                    dfs(i, total)  # 递归访问未访问过的节点
            valid[node] = 0

        for i in initial:
            judge = [False]  # 记录当前遍历的节点是否包含初始感染节点
            total = [0, False]  # [传播数量, 是否包含初始感染节点]
            dfs(i, total)

            if judge[0] == False:
                result.append([i, total[0]])  # 如果没有传播到其他初始感染节点，保存传播数量
            else:
                result.append([i, 1])  # 如果传播到了其他初始感染节点，传播数量为1
        # 按照第二个元素递减，若相等再按第一个元素递增排序
        result.sort(key=lambda x: (-x[1], x[0]))
        print(result)
        return result[0][0]


# 测试
p = Solution()
print(p.minMalwareSpread([[1,1,0],[1,1,0],[0,0,1]], [0,1,2]))





