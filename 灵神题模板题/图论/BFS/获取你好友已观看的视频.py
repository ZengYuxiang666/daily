# 这题没有其他的什么特别的，但是他的排序过程要好好学习，特别是defaultdict(int),本质是一个字典{xxx:int}
from collections import deque, defaultdict
class Solution(object):
    def watchedVideosByFriends(self, watchedVideos, friends, id, level):
        # 构建图
        n = len(friends)
        graph = defaultdict(list)
        for i in range(n):
            for f in friends[i]:
                graph[i].append(f)
                graph[f].append(i)  # 因为是无向图，双向添加

        # BFS 找到指定 level 的所有人
        queue = deque([id])
        visited = [False] * n
        visited[id] = True
        current_level = 0
        people_at_level = []
        while queue:
            level_size = len(queue)
            if current_level == level:
                people_at_level = list(queue)
                break

            for _ in range(level_size):
                person = queue.popleft()
                for neighbor in graph[person]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            current_level += 1
        # 统计指定 level 的人观看的视频
        video_count = defaultdict(int)
        for person in people_at_level:
            for video in watchedVideos[person]:
                video_count[video] += 1

        # 对视频按频率排序，频率相同的按字母顺序排序
        sorted_videos = sorted(video_count.items(), key=lambda x: (x[1], x[0]))

        # 返回视频列表
        return [video for video, _ in sorted_videos]