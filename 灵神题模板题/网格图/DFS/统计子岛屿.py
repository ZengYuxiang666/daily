# nonlocal关键字
# https://leetcode.cn/problems/count-sub-islands
# DFS：用于探索所有可能的路径，常用于寻找所有解或进行回溯（如解决迷宫、组合问题等）。DFS也可以用来找到路径，但不保证是最短路径。
# BFS：通常用于求解最短路径、最小步数等问题，因为BFS会层层推进，首先找到的解一定是最短路径。
class Solution(object):
    def countSubIslands(self, grid1, grid2):
        move_x = [1,-1,0,0]
        move_y = [0,0,1,-1]
        result = True
        def dfs(x,y):
            grid2[x][y] = 0
            nonlocal result
            if grid1[x][y] != 1:
                result = False
            for i in range(4):
                x1 = move_x[i]+x
                y1 = move_y[i]+y
                if 0<=x1<=len(grid2)-1 and 0<=y1<=len(grid2[0])-1:
                    if grid2[x1][y1] == 1:
                        dfs(x1,y1)
        total = 0
        for i in range(len(grid2)):
            for j in range(len(grid2[0])):
                if grid2[i][j] == 1:
                    result = True
                    dfs(i,j)
                    if result == True:
                        total += 1
        return total