import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator
import os

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
              '淘宝村数量（个）', '企业拥有网站数(个)','发明专利申请受理量(项)']

# 地区名称
regions = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽',
           '福建', '江西', '山东', '河南', '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
           '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']
# 输出文件路径
output_file = 'madate2.xlsx'

# 根据不同指标的特性选择不同的异常值检测方法
# Z-Score适用于符合正态分布的数据，主要用于百分比或比率数据：
z_score_cols = ['文教娱乐支出（%）', '农村居民恩格尔系数（%）', '第一产业/GDP（%）', 'GDP增长率（%）',
                '粮食总产量（万吨）/农作物总播种面积(千公顷)', '农用化肥施用量（万吨）/农作物总播种面积(千公顷)',
                '农药使用量（吨）/农作物总播种面积(千公顷)', 'R&D人员全时当量（人）/R&D经费支出(万元)']

# IQR用于离散数据或单位较大的数据，例如经济、财政、产业类数据：
iqr_cols = ['文化站个数（个）', '中国美丽休闲乡村数量', '旅行社数', '休闲农业与乡村旅游示范县(个)',
            '淘宝村数量（个）', '中大型拖拉机年末拥有量（万台）', '农用塑料薄膜使用量(吨)',
            '乡村办水电站数量', '农村用电量（万千瓦时）', '柴油使用量（万吨）', '发明专利申请受理量(项)',
            'R&D人员全时当量（人）', '移动电话年末用户(万户)', '企业拥有网站数(个)']

# LOF适用于时间序列数据，尤其是随时间变化的农业、生产类数据：
lof_cols = ['农用化肥施用量（万吨）', '农药使用量（吨）', '农作物总播种面积(千公顷)',
            '耕地灌溉面积（千公顷）','国家财政性教育经费(万元)']

# Isolation Forest适用于大范围经济、财政、产业总量类数据：
isolation_forest_cols = [ '农村居民家庭人均纯收入（元/人）', '农业保险密度（元/人）',
                         '农林牧渔业总产值（亿元）', 'R&D经费支出(万元)','粮食总产量（万吨）',]


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


# 3. LOF方法（局部异常因子）- 修改为适应小样本
def detect_outliers_lof(df, col):
    # 增加最小样本数要求
    if df[col].notnull().sum() < 3:  # LOF需要足够的数据点
        return df[col]

    # 对于小样本，采用较保守的异常检测参数
    values = df[col].dropna().values.reshape(-1, 1)
    n_neighbors = min(3, len(values) - 1)  # 减少邻居数量，更适合小样本
    if n_neighbors < 2:
        return df[col]

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)  # 提高contamination阈值
    outliers = lof.fit_predict(values)
    mask = (outliers == -1)  # 预测为异常点的索引

    # 对于非常小的样本，确认异常值不超过一定比例
    if sum(mask) > 0 and sum(mask) <= len(values) / 5:  # 异常值不超过1/5
        df.loc[df[col].dropna().index[mask], col] = np.nan

    return df[col]


# 4. Isolation Forest方法 - 修改为适应小样本
def detect_outliers_isolation_forest(df, col):
    # 增加最小样本数要求
    if df[col].notnull().sum() < 3:  # Isolation Forest需要足够的数据点
        return df[col]

    # 对小样本使用更保守的contamination值
    sample_size = df[col].notnull().sum()
    contamination = min(0.05, 1.0 / sample_size)  # 根据样本大小动态调整contamination

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=50)
    values = df[col].dropna().values.reshape(-1, 1)
    preds = model.fit_predict(values)
    mask = (preds == -1)  # 预测为异常点的索引

    # 确保不会标记太多点为异常
    if sum(mask) > 0 and sum(mask) <= len(values) / 5:  # 异常值不超过1/5
        df.loc[df[col].dropna().index[mask], col] = np.nan

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
ffill_bfill_cols = [ '农村居民家庭人均纯收入（元/人）', '农业保险密度（元/人）',
                    '农林牧渔业总产值（亿元）', '第一产业/GDP（%）',
                    '农用化肥施用量（万吨）', '农药使用量（吨）', '农作物总播种面积(千公顷)',
                    '耕地灌溉面积（千公顷）', '农村用电量（万千瓦时）', '柴油使用量（万吨）']

# 适合均值填补的方法
mean_cols = ['文化站个数（个）', '旅行社数', '休闲农业与乡村旅游示范县(个)',
              '中大型拖拉机年末拥有量（万台）', '农用塑料薄膜使用量(吨)',
             '乡村办水电站数量', '移动电话年末用户(万户)', '企业拥有网站数(个)','GDP增长率（%）']

# 适合中位数填补的方法
median_cols = ['发明专利申请受理量(项)', 'R&D人员全时当量（人）', 'R&D经费支出(万元)',
               'R&D人员全时当量（人）/R&D经费支出(万元)']

# 适合线性插值的方法
interpolate_cols = ['文教娱乐支出（%）', '农村居民恩格尔系数（%）', '粮食总产量（万吨）/农作物总播种面积(千公顷)',
                    '农用化肥施用量（万吨）/农作物总播种面积(千公顷)', '农药使用量（吨）/农作物总播种面积(千公顷)',
                    '国家财政性教育经费(万元)','中国美丽休闲乡村数量','淘宝村数量（个）','粮食总产量（万吨）']


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

            # 按“年份”升序排序
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


# 缺失值填补主函数
def fill_missing_values(df):
    """
    根据不同列的特性，使用不同的方法填补缺失值。
    """
    # 确保数据按年份排序
    df = df.sort_values('年份')

    # 应用各种填补方法
    df = fill_ffill_bfill(df, ffill_bfill_cols)
    df = fill_mean(df, mean_cols)
    df = fill_median(df, median_cols)
    df = fill_interpolate(df, interpolate_cols)

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


# 完善的process_region_data函数
def process_region_data(region_data, region_name):
    """
    处理单个地区的数据，包括异常值检测和缺失值填补。

    参数:
    region_data: 单个地区的DataFrame
    region_name: 地区名称

    返回:
    处理后的DataFrame
    """
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

    # 3. 计算派生指标（如果原始数据中不存在）
    # 确保所有需要用于计算的列都是数值类型
    for col in processed_data.columns:
        if col in ['粮食总产量（万吨）', '农作物总播种面积(千公顷)', '农用化肥施用量（万吨）',
                   '农药使用量（吨）', 'R&D人员全时当量（人）', 'R&D经费支出(万元)']:
            try:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            except:
                pass

    # 检查是否需要计算比率指标
    if '粮食总产量（万吨）/农作物总播种面积(千公顷)' not in processed_data.columns:
        if ('粮食总产量（万吨）' in processed_data.columns) and ('农作物总播种面积(千公顷)' in processed_data.columns):
            processed_data['粮食总产量（万吨）/农作物总播种面积(千公顷)'] = (
                    processed_data['粮食总产量（万吨）'] / processed_data['农作物总播种面积(千公顷)']
            )

    if '农用化肥施用量（万吨）/农作物总播种面积(千公顷)' not in processed_data.columns:
        if ('农用化肥施用量（万吨）' in processed_data.columns) and (
                '农作物总播种面积(千公顷)' in processed_data.columns):
            processed_data['农用化肥施用量（万吨）/农作物总播种面积(千公顷)'] = (
                    processed_data['农用化肥施用量（万吨）'] / processed_data['农作物总播种面积(千公顷)']
            )

    if '农药使用量（吨）/农作物总播种面积(千公顷)' not in processed_data.columns:
        if ('农药使用量（吨）' in processed_data.columns) and ('农作物总播种面积(千公顷)' in processed_data.columns):
            processed_data['农药使用量（吨）/农作物总播种面积(千公顷)'] = (
                    processed_data['农药使用量（吨）'] / processed_data['农作物总播种面积(千公顷)']
            )

    if 'R&D人员全时当量（人）/R&D经费支出(万元)' not in processed_data.columns:
        if ('R&D人员全时当量（人）' in processed_data.columns) and ('R&D经费支出(万元)' in processed_data.columns):
            processed_data['R&D人员全时当量（人）/R&D经费支出(万元)'] = (
                    processed_data['R&D人员全时当量（人）'] / processed_data['R&D经费支出(万元)']
            )

    # 4. 缺失值填补
    processed_data = fill_missing_values(processed_data)

    # 5. 四舍五入处理特定列
    for col in round_cols:
        if col in processed_data.columns:
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                processed_data[col] = processed_data[col].round().astype('Int64')

    # 6. 数据验证 - 检查处理后的数据是否合理
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


def main():
    file_path = 'data2.xlsx'
    if not os.path.exists(file_path):
        print(f'文件 {file_path} 不存在！')
        return

    # 读取Excel文件
    try:
        data = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    if '年份' not in data.columns or '地区' not in data.columns:
        print("缺少 '年份' 或 '地区' 列，请检查 Excel 文件格式！")
        return


    # 转换年份
    try:
        data['年份'] = pd.to_datetime(data['年份'], format='%Y').dt.year
    except Exception as e:
        print(f"年份转换失败: {e}")
        # 尝试修复年份列
        try:
            data['年份'] = pd.to_numeric(data['年份'], errors='coerce')
            print("已将年份转换为数值")
        except:
            print("无法修复年份列，请检查数据格式")
            return

    all_data = []

    for region in regions:
        print(f"正在处理 {region} 数据...")
        region_data = data[data['地区'] == region]

        if not region_data.empty:
            try:
                processed_data = process_region_data(region_data.copy(), region)
                all_data.append(processed_data)
            except Exception as e:
                print(f"处理 {region} 数据时出错: {e}")
                continue  # 跳过错误的区域，继续处理其他区域

    if all_data:
        final_data = pd.concat(all_data, ignore_index=True)
        try:
            final_data.to_excel(output_file, index=False, engine='openpyxl')
            print(f"✅ 数据处理完成！已保存至 {output_file}")
        except Exception as e:
            print(f"保存Excel文件失败: {e}")
    else:
        print("没有成功处理任何数据！")


if __name__ == "__main__":
    main()