from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.interpolate import PchipInterpolator,Akima1DInterpolator
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np


# 指标名称
value_arr = ['文化站个数（个）', '文教娱\n乐支出（%）', '国家财政性教育经费(万元)',
             '农村居民家庭\n人均纯收入（元/人）', '农村居民恩格尔系数（%）', '农业保险密度（元/人）',
             '农林牧渔业总产值（亿元）',
             '第一产业/GDP（%）', '粮食总产量（万吨）', '农用化肥施用量（万吨）', '农药使用量（吨）',
             '农作物总播种面积(千公顷)',
             '粮食总产量（万吨）/农作物总播种面积(千公顷)', '农用化肥施用量（万吨）/农作物总播种面积(千公顷)',
             '农药使用量（吨）/农作物总播种面积(千公顷)', '耕地灌溉面积（千公顷）', '中国美丽休闲乡村数量', '旅行社数',
             '休闲农业与乡村旅游示范县(个)', '淘宝村数量（个）', '中大型拖拉机年末拥有量（万台）',
             '农用塑料薄膜使用量\n(吨)',
             '乡村办水电站数量', '农村用电量（万千瓦时）', '柴油使用量（万吨）', '发明专利申请受理量(项)',
             'R&D人员全时当量（人）',
             'R&D经费支出(万元)', 'R&D人员全时当量（人）/R&D经费支出(万元)', '移动电话年末用户(万户)',
             '企业拥有网站数(个)',
             'GDP增长率（%）']
# 需要四舍五入的指标
round_cols = ['文化站个数（个）', '旅行社数', '休闲农业与乡村旅游示范县(个)', '乡村办水电站数量', '中国美丽休闲乡村数量',
              '淘宝村数量（个）', '企业拥有网站数(个)', '发明专利申请受理量(项)']

# 地区名称
regions = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽',
           '福建', '江西', '山东', '河南', '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
           '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']
# 输出文件路径
output_file = 'madate2.xlsx'

# 由前面两个相除得到的指标
derive_cols = ['粮食总产量（万吨）/农作物总播种面积(千公顷)', '农用化肥施用量（万吨）/农作物总播种面积(千公顷)',
               '农药使用量（吨）/农作物总播种面积(千公顷)',
               '农药使用量（吨）/农作物总播种面积(千公顷)', 'R&D人员全时当量（人）/R&D经费支出(万元)',
               ]

# 根据不同指标的特性选择不同的异常值检测方法
# Z-Score适用于符合正态分布的数据，主要用于百分比或比率数据：
z_score_cols = ['文教娱乐支出（%）', '农村居民恩格尔系数（%）', '第一产业/GDP（%）', 'GDP增长率（%）', ]

# IQR用于离散数据或单位较大的数据，例如经济、财政、产业类数据：
iqr_cols = ['文化站个数（个）', '中国美丽休闲乡村数量', '旅行社数', '休闲农业与乡村旅游示范县(个)',
            '中大型拖拉机年末拥有量（万台）', '农用塑料薄膜使用量(吨)',
            '乡村办水电站数量', '农村用电量（万千瓦时）', '柴油使用量（万吨）', '发明专利申请受理量(项)',
            'R&D人员全时当量（人）', '移动电话年末用户(万户)', '企业拥有网站数(个)']

# LOF适用于时间序列数据，尤其是随时间变化的农业、生产类数据：
lof_cols = ['农用化肥施用量（万吨）', '农药使用量（吨）', '农作物总播种面积(千公顷)',
            '耕地灌溉面积（千公顷）', '国家财政性教育经费(万元)','淘宝村数量（个）']

# Isolation Forest适用于大范围经济、财政、产业总量类数据：
isolation_forest_cols = ['农村居民家庭人均纯收入（元/人）', '农业保险密度（元/人）',
                         '农林牧渔业总产值（亿元）', 'R&D经费支出(万元)', '粮食总产量（万吨）']


# 1. Z-Score方法（适用于比率、百分比数据）- 修改为适应小样本
def detect_outliers_zscore(df, col):
    if df[col].notnull().sum() < 3:  # 样本太少，无法可靠检测
        return df[col]

    # 对于小样本，放宽阈值条件
    threshold = 1.5  # 放宽Z-score阈值
    z_scores = np.abs(zscore(df[col].dropna(), nan_policy='omit'))  # 使用nan_policy处理NaN
    outlier_indices = df[col].dropna().index[z_scores > threshold]

    # 对于小样本，记录异常值而不是直接替换为NaN
    if len(outlier_indices) > 0 and len(outlier_indices) < len(df[col].dropna()) / 3:  # 异常值不超过1/3
        df.loc[outlier_indices, col] = np.nan

    return df[col]


# 2. IQR方法（四分位数法）- 修改为适应小样本
def detect_outliers_iqr(df, col):
    if df[col].notnull().sum() < 4:  # 样本太少，无法可靠计算四分位数
        return df[col]

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    # 放宽阈值，从1.5扩大到2.0
    lower_bound = q1 - 2.0 * iqr
    upper_bound = q3 + 2.0 * iqr

    # 对于小样本，仅处理极端异常值
    extreme_outliers = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if len(extreme_outliers) > 0 and len(extreme_outliers) < len(df[col].dropna()) / 4:  # 异常值不超过1/4
        df.loc[extreme_outliers, col] = np.nan

    return df[col]


# 3. LOF方法（局部异常因子）
def detect_outliers_lof(df, col):
    if df[col].notnull().sum() < 3:  # LOF需要足够的数据点
        return df[col]

    # 对数据添加微小的随机噪声来打破重复值
    values = df[col].dropna().values

    # 检查是否存在重复值
    if len(np.unique(values)) < len(values):
        # 添加微小随机噪声以打破重复值（噪声比例为数据标准差的0.01%）
        noise_scale = np.std(values) * 0.0001 if np.std(values) > 0 else 0.0001
        values = values + np.random.normal(0, noise_scale, size=values.shape)

    values = values.reshape(-1, 1)

    n_neighbors = min(3, len(values) - 1)  # 减少邻居数量，更适合小样本
    if n_neighbors < 2:
        return df[col]

    try:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
        outliers = lof.fit_predict(values)
        mask = (outliers == -1)  # 预测为异常点的索引

        # 对于非常小的样本，确认异常值不超过一定比例
        if sum(mask) > 0 and sum(mask) <= len(values) / 10:  # 异常值不超过1/10
            df.loc[df[col].dropna().index[mask], col] = np.nan
    except Exception as e:
        print(f"LOF异常检测出错（列：{col}）：{e}")

    return df[col]


def detect_outliers_isolation_forest(df, col):
    """
    使用 Isolation Forest 检测异常值
    """
    if df[col].notnull().sum() < 3:  # 确保样本数足够
        return df[col]

    sample_size = df[col].notnull().sum()
    contamination = min(0.02, 1.0 / (sample_size * 2))  # 进一步降低 contamination

    # 构建 Isolation Forest 模型
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,  # 增加基学习器数量，提高稳定性
        max_samples="auto"  # 让模型自适应样本量
    )

    values = df[col].dropna().values.reshape(-1, 1)
    preds = model.fit_predict(values)  # 预测异常值
    scores = model.decision_function(values)  # 获取异常分数

    # 动态设定更宽松的阈值（排除得分极端的点）
    threshold = np.percentile(scores, 2)  # 仅排除最极端 2% 的点
    mask = scores < threshold  # 判定异常值

    if sum(mask) > 0 and sum(mask) <= len(values) / 10:  # 限制最多 10% 为异常值
        df.loc[df[col].dropna().index[mask], col] = np.nan  # 只剔除少数异常点

    return df[col]

# 异常值检测主函数
def abnormal_value_detection_method(df):
    for col in df.columns:
        if col in ['年份', '地区']:  # 跳过非数值列
            continue

        if col in z_score_cols:
            df[col] = detect_outliers_zscore(df, col)
        elif col in iqr_cols:
            df[col] = detect_outliers_iqr(df, col)
        elif col in lof_cols:
            df[col] = detect_outliers_lof(df, col)
        elif col in isolation_forest_cols:
            df[col] = detect_outliers_isolation_forest(df, col)
    return df


# 根据不同指标的特性选择不同的填补方法
# 适合 前向填充（ffill）和后向填充（bfill）的方法
ffill_bfill_cols = ['农村居民家庭人均纯收入（元/人）', '农业保险密度（元/人）',
                    '农林牧渔业总产值（亿元）', '第一产业/GDP（%）',
                    '农用化肥施用量（万吨）', '农药使用量（吨）', '农作物总播种面积(千公顷)',
                    '耕地灌溉面积（千公顷）', '农村用电量（万千瓦时）', '柴油使用量（万吨）']

# 适合均值填补的方法
mean_cols = ['文化站个数（个）', '休闲农业与乡村旅游示范县(个)',
             '中大型拖拉机年末拥有量（万台）', '农用塑料薄膜使用量(吨)',
             '乡村办水电站数量', '移动电话年末用户(万户)', '企业拥有网站数(个)', 'GDP增长率（%）']

# 适合中位数填补的方法
median_cols = ['发明专利申请受理量(项)', 'R&D人员全时当量（人）', 'R&D经费支出(万元)']

# 适合 PCHIP 插值器的方法
interpolate_cols = ['文教娱乐支出（%）', '农村居民恩格尔系数（%）', '国家财政性教育经费(万元)', '中国美丽休闲乡村数量',
                     '粮食总产量（万吨）']

# 适合Akima 插值 或 样条插值的方法
akima_cols = ['淘宝村数量（个）','旅行社数']



# 填补方法 - 针对小样本的修改版本
def fill_ffill_bfill(df, cols):
    """
    对指定列使用前向填充（ffill）和后向填充（bfill），适合时间序列数据。
    """
    for col in cols:
        if col in df.columns and df[col].isna().any():
            # 确保数据按年份排序
            df = df.sort_values('年份')
            # 首先尝试线性插值（对于内部缺失值）
            df[col] = df[col].interpolate(method='linear', limit_direction='both', limit=2)
            # 然后使用前向和后向填充处理剩余缺失值
            df[col] = df[col].ffill().bfill()
    return df


def fill_mean(df, cols):
    """
    对指定列使用均值填补 NaN 值。
    适用于小样本时间序列，优先用滚动窗口均值，最后才用全局均值兜底。
    """
    df = df.copy()

    for col in cols:
        if col in df.columns and df[col].isna().any():
            # 按年份排序
            df = df.sort_values('年份')

            # 非缺失值数量
            non_null_count = df[col].notnull().sum()

            if non_null_count < 3:
                # 如果非缺失值太少，直接用整体均值
                df[col] = df[col].fillna(df[col].mean())
            else:
                # 使用3年滚动窗口均值（避免只用前一个值）
                rolling_mean = df[col].rolling(window=3, min_periods=2, center=True).mean()

                # 先填补滚动均值
                df[col] = df[col].fillna(rolling_mean)

                # 对仍然是 NaN 的数据，使用全局均值
                df[col] = df[col].fillna(df[col].mean())

    return df


def fill_median(df, cols):
    """
    对指定列使用中位数填补 NaN 值，对极端值敏感的数据更稳健。
    """
    for col in cols:
        if col in df.columns and df[col].isna().any():
            # 如果非缺失值太少，直接使用全局中位数
            if df[col].notnull().sum() < 3:
                df[col] = df[col].fillna(df[col].median())
            else:
                # 使用临近年份的中位数
                df = df.sort_values('年份')
                # 滑动窗口中位数
                df[col] = df[col].fillna(df[col].rolling(window=3, min_periods=1, center=True).median())
                # 对于仍然缺失的值，使用全局中位数
                df[col] = df[col].fillna(df[col].median())
    return df

def fill_interpolate(df, cols, method='pchip'):
    """
    对指定列使用插值方法填补 NaN 值，适合随时间变化平稳的数据。
    优化：使用 scipy 的 PchipInterpolator 进行平滑的三次插值，避免边界震荡。
    如果失败则退回到线性插值，最后用前向/后向填补兜底。
    """
    df = df.copy()

    for col in cols:
        if col in df.columns and df[col].isna().any():
            # 保存原始的 NaN 位置
            original_nan_mask = df[col].isna()

            # 按"年份"升序排序
            df = df.sort_values('年份')

            x = df['年份'].values
            y = df[col].values

            valid_mask = ~np.isnan(y)
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            non_null_count = valid_mask.sum()

            if non_null_count >= 4:
                try:
                    # 使用 PCHIP 插值器（保持平滑，防止边界震荡）
                    interpolator = PchipInterpolator(x_valid, y_valid, extrapolate=True)
                    y_interp = y.copy()
                    nan_mask = np.isnan(y)
                    y_interp[nan_mask] = interpolator(x[nan_mask])
                    df[col] = y_interp
                except Exception as e:
                    print(f"列 {col} 在 PCHIP 插值时出错：{e}，将使用线性插值")
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
            else:
                # 数据点不足时使用线性插值
                df[col] = df[col].interpolate(method='linear', limit_direction='both')

            # 检查仍然存在 NaN 的地方，使用 ffill / bfill
            remaining_nan_mask = df[col].isna()
            if remaining_nan_mask.any():
                temp_series = df[col].copy().ffill().bfill()
                df.loc[remaining_nan_mask, col] = temp_series[remaining_nan_mask]
                ffill_bfill_count = remaining_nan_mask.sum()
                print(f"警告: 在列 {col} 中有 {ffill_bfill_count} 个值无法通过插值法填补，已用前向/后向填充。")

    return df


from scipy.interpolate import CubicSpline


def fill_interpolate_robust(df, cols):
    """
    对指定列使用插值方法填补 NaN 值，适用于随时间变化大的数据。
    - 优先使用 Akima 插值（更适应突变）
    - 失败时回退到 CubicSpline 插值
    - 对于插值后仍然缺失的值，根据历史增长率进行填充
    """
    df = df.copy()

    for col in cols:
        if col in df.columns and df[col].isna().any():
            # 记录 NaN 位置
            original_nan_mask = df[col].isna()

            # 按"年份"升序排序
            df = df.sort_values('年份')

            x = df['年份'].values
            y = df[col].values

            valid_mask = ~np.isnan(y)
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            non_null_count = valid_mask.sum()

            if non_null_count >= 4:
                try:
                    # 先尝试 Akima 插值（适用于数据变化较大）
                    interpolator = Akima1DInterpolator(x_valid, y_valid)
                    y_interp = y.copy()
                    nan_mask = np.isnan(y)
                    y_interp[nan_mask] = interpolator(x[nan_mask])
                    df[col] = y_interp
                    print(f"列 {col} 使用 Akima 插值填充完成")
                except Exception as e:
                    print(f"列 {col} 在 Akima 插值时出错：{e}，改用 CubicSpline")
                    try:
                        # 使用 CubicSpline 插值，设置 extrapolate=True 允许外推
                        interpolator = CubicSpline(x_valid, y_valid, extrapolate=True)
                        y_interp = y.copy()
                        nan_mask = np.isnan(y)
                        y_interp[nan_mask] = interpolator(x[nan_mask])
                        df[col] = y_interp
                        print(f"列 {col} 使用 CubicSpline 插值填充完成")
                    except Exception as e2:
                        print(f"列 {col} 在 CubicSpline 插值时出错：{e2}，改用 pandas 内置插值")
                        # 回退到 pandas 内置的三次样条插值
                        df[col] = df[col].interpolate(method='cubic', limit_direction='both')
            else:
                # 数据点不足时使用 pandas 内置的三次样条插值
                print(f"列 {col} 有效数据点不足4个，使用 pandas 内置的 cubic 插值")
                df[col] = df[col].interpolate(method='cubic', limit_direction='both')

            # 检查是否仍有 NaN 值（可能出现在开头或结尾）
            if df[col].isna().sum() > 0:
                print(f"列 {col} 在插值后仍有 {df[col].isna().sum()} 个 NaN 值，使用增长率方法填充")

                # 基于增长率的填充
                # 计算最近几个有效值的平均增长率
                temp_df = df.copy()
                temp_df['previous'] = temp_df[col].shift(1)
                temp_df['growth_rate'] = (temp_df[col] / temp_df['previous']) - 1

                # 计算最近3年平均增长率（如果有足够的数据）
                if temp_df['growth_rate'].dropna().shape[0] >= 2:
                    avg_growth_rate = temp_df['growth_rate'].dropna().tail(3).mean()

                    # 处理开头的缺失值（如果有）
                    if df[col].isna().iloc[0]:
                        # 找到第一个非NaN值的位置
                        first_valid_idx = df[col].first_valid_index()
                        first_valid_value = df.loc[first_valid_idx, col]
                        first_valid_year = df.loc[first_valid_idx, '年份']

                        # 向前填充
                        for idx in df.index:
                            if df.loc[idx, '年份'] < first_valid_year and np.isnan(df.loc[idx, col]):
                                years_diff = first_valid_year - df.loc[idx, '年份']
                                # 使用逆增长率向前推
                                df.loc[idx, col] = first_valid_value / ((1 + avg_growth_rate) ** years_diff)

                    # 处理末尾的缺失值（如果有）
                    if df[col].isna().iloc[-1]:
                        # 找到最后一个非NaN值的位置
                        last_valid_idx = df[col].last_valid_index()
                        last_valid_value = df.loc[last_valid_idx, col]
                        last_valid_year = df.loc[last_valid_idx, '年份']

                        # 向后填充
                        for idx in df.index:
                            if df.loc[idx, '年份'] > last_valid_year and np.isnan(df.loc[idx, col]):
                                years_diff = df.loc[idx, '年份'] - last_valid_year
                                # 使用增长率向后推
                                df.loc[idx, col] = last_valid_value * ((1 + avg_growth_rate) ** years_diff)
                else:
                    # 如果增长率数据不足，使用常数增长率（如3%）
                    constant_growth_rate = 0.03
                    print(
                        f"警告：列 {col} 的有效数据点不足以计算可靠的增长率，使用默认增长率 {constant_growth_rate * 100}%")

                    # 应用同样的前后填充逻辑，但使用常数增长率
                    # 处理开头的缺失值
                    if df[col].isna().iloc[0]:
                        first_valid_idx = df[col].first_valid_index()
                        first_valid_value = df.loc[first_valid_idx, col]
                        first_valid_year = df.loc[first_valid_idx, '年份']

                        for idx in df.index:
                            if df.loc[idx, '年份'] < first_valid_year and np.isnan(df.loc[idx, col]):
                                years_diff = first_valid_year - df.loc[idx, '年份']
                                df.loc[idx, col] = first_valid_value / ((1 + constant_growth_rate) ** years_diff)

                    # 处理末尾的缺失值
                    if df[col].isna().iloc[-1]:
                        last_valid_idx = df[col].last_valid_index()
                        last_valid_value = df.loc[last_valid_idx, col]
                        last_valid_year = df.loc[last_valid_idx, '年份']

                        for idx in df.index:
                            if df.loc[idx, '年份'] > last_valid_year and np.isnan(df.loc[idx, col]):
                                years_diff = df.loc[idx, '年份'] - last_valid_year
                                df.loc[idx, col] = last_valid_value * ((1 + constant_growth_rate) ** years_diff)

    return df

# 计算派生指标
def calculate_derived_indicators(df):
    result_df = df.copy()

    # Check and calculate grain yield per area if columns exist
    if '粮食总产量（万吨）' in df.columns and '农作物总播种面积(千公顷)' in df.columns:
        result_df['粮食总产量（万吨）/农作物总播种面积(千公顷)'] = (
                df['粮食总产量（万吨）'] / df['农作物总播种面积(千公顷)']
        )

    # Check and calculate fertilizer usage per area if columns exist
    if '农用化肥施用量（万吨）' in df.columns and '农作物总播种面积(千公顷)' in df.columns:
        result_df['农用化肥施用量（万吨）/农作物总播种面积(千公顷)'] = (
                df['农用化肥施用量（万吨）'] / df['农作物总播种面积(千公顷)']
        )

    # Check and calculate pesticide usage per area if columns exist
    if '农药使用量（吨）' in df.columns and '农作物总播种面积(千公顷)' in df.columns:
        result_df['农药使用量（吨）/农作物总播种面积(千公顷)'] = (
                df['农药使用量（吨）'] / df['农作物总播种面积(千公顷)']
        )

    # Check and calculate R&D personnel per R&D expenditure if columns exist
    if 'R&D人员全时当量（人）' in df.columns and 'R&D经费支出(万元)' in df.columns:
        result_df['R&D人员全时当量（人）/R&D经费支出(万元)'] = (
                df['R&D人员全时当量（人）'] / df['R&D经费支出(万元)']
        )

    return result_df


# 处理负值
def fix_negative_values(df):
    result_df = df.copy()

    # 遍历所有列
    for col in result_df.columns:
        if col not in ['年份', '地区', 'GDP增长率（%）']:  # 排除年份、地区列和GDP增长率
            # 确保列是数值类型
            if pd.api.types.is_numeric_dtype(result_df[col]):
                # 将负值替换为0
                negative_mask = result_df[col] < 0
                if negative_mask.any():
                    neg_count = negative_mask.sum()
                    print(f"在列 {col} 中发现 {neg_count} 个负值，已替换为0")
                    result_df.loc[negative_mask, col] = 0

    return result_df


# 缺失值填补主函数
def fill_missing_values(df):
    df = df.sort_values('年份')

    # 应用各种填补方法
    df = fill_ffill_bfill(df, ffill_bfill_cols)
    df = fill_mean(df, mean_cols)
    df = fill_median(df, median_cols)
    df = fill_interpolate(df, interpolate_cols)
    df = fill_interpolate_robust(df,akima_cols)

    # 处理任何剩余的缺失值（使用简单插值或最近邻值）
    for col in df.columns:
        if col not in ['年份', '地区'] and df[col].isna().any():
            # 尝试使用简单的线性插值
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            # 如果仍有缺失值，使用前向/后向填充
            df[col] = df[col].ffill().bfill()

    # 对需要四舍五入的指标进行处理
    for col in round_cols:
        if col in df.columns:
            # 确保值是数值类型
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round().astype('Int64')  # 使用Int64可以处理NaN

    return df

def process_region_data(region_data, region_name):
    print(f'处理中: {region_name} ...')

    # 确保年份为日期格式，并只保留年份数值
    region_data['年份'] = pd.to_datetime(region_data['年份'], format='%Y').dt.year

    # 拷贝数据，避免原始数据被修改
    processed_data = region_data.copy()

    # 确保'地区'列存在
    if '地区' in processed_data.columns:
        processed_data['地区'] = region_name
    else:
        processed_data.insert(0, '地区', region_name)  # 添加地区列

    # 1. 清洗数据 - 处理明显错误的数据（例如负值等）
    for col in processed_data.columns:
        if col not in ['年份', '地区']:
            # 先尝试转换为数值类型
            try:
                # 如果列是对象类型，尝试转换为数值
                if pd.api.types.is_object_dtype(processed_data[col]):
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            except Exception as e:
                print(f"无法将列 {col} 转换为数值类型: {e}")
                continue  # 跳过这一列的处理

            # 对于比率类指标，确保在合理范围内
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                if '（%）' in col:
                    processed_data.loc[processed_data[col] > 100, col] = np.nan
                    processed_data.loc[processed_data[col] < 0, col] = np.nan

                # 对于非负指标，确保值不为负
                if col not in ['GDP增长率（%）']:  # GDP增长率可以为负
                    processed_data.loc[processed_data[col] < 0, col] = np.nan

    # 2. 异常值检测
    processed_data = abnormal_value_detection_method(processed_data)

    # 3. 确保所有需要用于计算的列都是数值类型
    for col in processed_data.columns:
        if col in ['粮食总产量（万吨）', '农作物总播种面积(千公顷)', '农用化肥施用量（万吨）',
                   '农药使用量（吨）', 'R&D人员全时当量（人）', 'R&D经费支出(万元)']:
            try:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            except:
                pass

    # 使用函数计算派生指标
    processed_data = calculate_derived_indicators(processed_data)

    # 4. 缺失值填补
    processed_data = fill_missing_values(processed_data)

    # 5. 处理所有非GDP增长率的负值，替换为0
    processed_data = fix_negative_values(processed_data)

    # 6. 四舍五入处理特定列
    for col in round_cols:
        if col in processed_data.columns:
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                processed_data[col] = processed_data[col].round().astype('Int64')

    # 7. 数据验证 - 检查处理后的数据是否合理
    for col in processed_data.columns:
        if col not in ['年份', '地区']:
            # 检查是否存在无限值或极端值
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                processed_data[col] = processed_data[col].replace([np.inf, -np.inf], np.nan)

            # 如果某列处理后全部为NaN，输出警告
            if processed_data[col].isna().all() and len(processed_data) > 0:
                print(f"警告: {region_name} 的 {col} 列全部为NaN!")

    print(f'{region_name} 处理完成!')
    return processed_data
