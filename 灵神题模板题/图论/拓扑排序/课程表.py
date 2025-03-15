# 只有入度为0的节点才能进入queue队列，才能进行拓扑排序，因为只有为0时他的前置条件才全实现了
# 当有环时，两个节点的入度就不可能变成0了，就不可能将节点加入queue中，就更不可能造成死循环了，所以只需要判断拓扑排序后的结果的长度就能判断有没有环了
# https://leetcode.cn/problems/course-schedule-ii
from collections import deque, defaultdict
class Solution:
    def findOrder(self, numCourses: int, prerequisites):
        # 1. 构建图的邻接表和入度数组
        graph = defaultdict(list)
        indegree = [0] * numCourses  # 节点的入度数量

        for course, prereq in prerequisites:
            graph[prereq].append(course)
            indegree[course] += 1

        # 2. 找到所有入度为 0 的课程
        queue = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                queue.append(i)

        result = []

        # 3. 执行拓扑排序
        while queue:
            course = queue.popleft()
            result.append(course)

            # 遍历该课程的所有后续课程
            for next_course in graph[course]:
                indegree[next_course] -= 1
                if indegree[next_course] == 0:
                    queue.append(next_course)

        # 4. 判断是否能够完成所有课程
        if len(result) == numCourses:
            return result
        else:
            return []  # 如果有环，返回空数组