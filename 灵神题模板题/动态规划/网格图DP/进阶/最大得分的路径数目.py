# 最核心的思路是在遍历每个节点是定义了max_score和ways，每次从3个方向来更新这两个数据，最后储存在dp数组中
# https://leetcode.cn/problems/number-of-paths-with-max-score
def pathsWithMaxScore(board):
    MOD = 10 ** 9 + 7
    n = len(board)

    # 初始化 dp 数组，dp[i][j] = (score, count)
    dp = [[(0, 0)] * n for _ in range(n)]

    # 起点 'E' 位置
    dp[0][0] = (0, 1)

    # 正序遍历 DP 计算
    for i in range(n):
        for j in range(n):
            if board[i][j] == 'X' or (i == 0 and j == 0):  # 障碍或起点，跳过
                continue

            max_score, ways = 0, 0

            # 可能的来源（左、上、左上）
            for x, y in [(i - 1, j), (i, j - 1), (i - 1, j - 1)]:
                if 0 <= x < n and 0 <= y < n and dp[x][y][1] > 0:
                    score = dp[x][y][0]
                    if score > max_score:
                        max_score = score
                        ways = dp[x][y][1]
                    elif score == max_score:
                        ways = (ways + dp[x][y][1]) % MOD

            # 计算当前格子得分
            if board[i][j] != 'S':
                max_score += int(board[i][j])

            # 更新 DP 结果
            dp[i][j] = (max_score, ways)

    return list(dp[n - 1][n - 1]) if dp[n - 1][n - 1][1] > 0 else [0, 0]
