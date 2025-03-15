# 将问题转换成图有没有环的问题，这个判断图有没有环的dfs函数需要记住
# https://leetcode.cn/problems/course-schedule
from typing import List


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 构建邻接表
        graph = {i: [] for i in range(numCourses)}
        for course, pre in prerequisites:
            graph[pre].append(course)

        # 记录节点的访问状态: 0 = 未访问, 1 = 正在访问, 2 = 已访问
        visited = [0] * numCourses

        # DFS 判断是否有环
        def dfs(course):
            if visited[course] == 1:  # 发现环
                return False
            if visited[course] == 2:  # 已经处理过，直接返回
                return True

            # 标记当前课程正在访问
            visited[course] = 1
            for next_course in graph[course]:
                if not dfs(next_course):
                    return False

            # 完成当前课程的DFS，标记为已访问
            visited[course] = 2
            return True

        # 对所有课程进行DFS遍历
        for course in range(numCourses):
            if visited[course] == 0:  # 如果课程未访问过，进行DFS
                if not dfs(course):
                    return False

        return True