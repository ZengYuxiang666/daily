import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import os

# ✅ 指标名称
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
# ✅ 需要四舍五入的指标
round_cols = ['文化站个数（个）', '旅行社数', '休闲农业与乡村旅游示范县(个)', '乡村办水电站数量', '中国美丽休闲乡村数量',
              '淘宝村数量（个）', '企业拥有网站数(个)']

# ✅ 地区名称
regions = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽',
           '福建', '江西', '山东', '河南', '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
           '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']
# ✅ 输出文件路径
output_file = 'madate2.xlsx'

# 为小样本数据调整异常值检测方法分类
# 所有指标使用最为保守的方法检测异常值
percent_cols = ['文教娱乐支出（%）', '农村居民恩格尔系数（%）', '第一产业/GDP（%）', 'GDP增长率（%）',
                '粮食总产量（万吨）/农作物总播种面积(千公顷)', '农用化肥施用量（万吨）/农作物总播种面积(千公顷)',
                '农药使用量（吨）/农作物总播种面积(千公顷)', 'R&D人员全时当量（人）/R&D经费支出(万元)']

count_cols = ['文化站个数（个）', '休闲农业与乡村旅游示范县(个)',
             '中大型拖拉机年末拥有量（万台）', '农用塑料薄膜使用量(吨)',
               '发明专利申请受理量(项)', 'R&D人员全时当量（人）', '企业拥有网站数(个)']

amount_cols = ['粮食总产量（万吨）', '农用化肥施用量（万吨）', '农药使用量（吨）', '农作物总播种面积(千公顷)',
               '耕地灌溉面积（千公顷）', '农村用电量（万千瓦时）', '柴油使用量（万吨）', '移动电话年末用户(万户)']

finance_cols = ['国家财政性教育经费(万元)', '农村居民家庭人均纯收入（元/人）', '农业保险密度（元/人）',
                '农林牧渔业总产值（亿元）', 'R&D经费支出(万元)','旅行社数','中国美丽休闲乡村数量','淘宝村数量（个）','乡村办水电站数量']


def convert_to_numeric(series):
    """将Series转换为数值型，处理特殊值如'--'"""
    # 创建一个新的Series来存储结果
    numeric_series = pd.Series(index=series.index, dtype=float)

    for idx, val in series.items():
        try:
            if isinstance(val, str):
                if val == '--' or val.strip() == '':
                    numeric_series[idx] = np.nan
                else:
                    # 尝试将字符串转换为数值
                    numeric_series[idx] = float(val.replace(',', ''))  # 处理可能的千位分隔符
            else:
                numeric_series[idx] = float(val) if not pd.isna(val) else np.nan
        except (ValueError, TypeError):
            print(f"无法转换值 '{val}' 为数值，已设置为NaN")
            numeric_series[idx] = np.nan

    return numeric_series


def detect_outliers_basic(df, col, method='iqr', threshold_multiplier=2.0):
    """
    基本的异常值检测方法，适用于小样本

    参数:
    - df: 数据框
    - col: 列名
    - method: 使用的方法 ('iqr' 或 'zscore')
    - threshold_multiplier: 阈值倍数，对于小样本应当放宽（如2.0而非1.5）
    """
    # 首先转换为数值型
    series = convert_to_numeric(df[col])

    if series.notnull().sum() <= 3:  # 如果样本太小，不进行异常值处理
        df[col] = series
        return df[col]

    try:
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold_multiplier * iqr
            upper_bound = q3 + threshold_multiplier * iqr

            # 获取异常值索引
            outlier_indices = series.index[(series.notnull()) &
                                           ((series < lower_bound) | (series > upper_bound))]
        elif method == 'zscore':
            # 对于小样本，使用更宽松的Z分数阈值
            z_scores = np.abs((series - series.mean()) / series.std())
            outlier_indices = series.index[(series.notnull()) & (z_scores > 3.0)]
        else:
            raise ValueError(f"不支持的方法: {method}")

        # 设置异常值为NaN
        if len(outlier_indices) > 0:
            # 对于小样本，打印异常值便于人工检查
            print(f"在列 {col} 中检测到 {len(outlier_indices)} 个异常值")
            for idx in outlier_indices:
                print(f"  - 索引 {idx}, 值 {series[idx]}")
            series.loc[outlier_indices] = np.nan

        df[col] = series
        return df[col]
    except Exception as e:
        print(f"处理{col}时出错: {e}")
        df[col] = series
        return df[col]


def fill_missing_values(df, col, years_col='年份', method='interpolate'):
    """
    根据时间序列特性填补缺失值

    参数:
    - df: 数据框
    - col: 需要填补的列
    - years_col: 年份列名
    - method: 填补方法
    """
    # 确保数据为数值型
    df[col] = convert_to_numeric(df[col])

    try:
        if method == 'interpolate':
            # 对于小样本，线性插值通常是最佳选择
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
        elif method == 'ffill_bfill':
            # 前向后向填充，适合连续变化的数据
            df[col] = df[col].ffill().bfill()
        elif method == 'mean':
            # 使用均值填充
            if df[col].notnull().sum() > 0:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
        elif method == 'median':
            # 使用中位数填充
            if df[col].notnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        elif method == 'time_weighted':
            # 对于小样本，使用时间加权平均
            if df[col].notnull().sum() >= 2:
                # 确保年份是数值
                years = pd.to_numeric(df[years_col], errors='coerce')
                # 丢弃无效年份的行
                valid_data = df.dropna(subset=[years_col, col])

                if len(valid_data) >= 2:
                    # 创建插值函数
                    try:
                        interp_func = interp1d(
                            valid_data[years_col].values,
                            valid_data[col].values,
                            bounds_error=False,
                            fill_value='extrapolate'
                        )

                        # 对所有NaN值进行填补
                        for idx in df.index[df[col].isna()]:
                            if not pd.isna(df.loc[idx, years_col]):
                                df.loc[idx, col] = interp_func(df.loc[idx, years_col])
                    except:
                        # 如果插值失败，回退到简单插值
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')

        # 如果还有缺失值，使用前向/后向填充
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()

        return df
    except Exception as e:
        print(f"填补{col}缺失值时出错: {e}")
        return df


def process_region_data(region_data, region_name):
    """
    处理单个地区的数据，包括异常值检测和空缺值填补，针对小样本进行优化
    """
    print(f'🚀 处理中: {region_name} ...')
    try:
        # 1. 确保年份为数值格式
        region_data['年份'] = pd.to_datetime(region_data['年份'], format='%Y', errors='coerce').dt.year

        # 如果年份列存在问题，尝试直接转换
        if region_data['年份'].isna().any():
            region_data['年份'] = region_data['年份'].apply(
                lambda x: int(x) if isinstance(x, (int, float)) else (
                    int(x) if isinstance(x, str) and x.isdigit() else np.nan
                )
            )

        # 按年份排序，便于时间序列分析
        region_data = region_data.sort_values('年份')

        # 检查每个列是否存在
        existing_columns = set(region_data.columns)

        # 2. 异常值检测 - 对于小样本使用更保守的方法

        # 对百分比类数据使用Z-Score方法
        for col in percent_cols:
            if col in existing_columns:
                region_data[col] = detect_outliers_basic(region_data, col, method='zscore', threshold_multiplier=3.0)

        # 对计数类数据使用IQR方法，但阈值设置更宽松
        for col in count_cols:
            if col in existing_columns:
                region_data[col] = detect_outliers_basic(region_data, col, method='iqr', threshold_multiplier=2.0)

        # 对数量和金融类数据使用IQR方法，但阈值更为宽松
        for col in amount_cols + finance_cols:
            if col in existing_columns:
                region_data[col] = detect_outliers_basic(region_data, col, method='iqr', threshold_multiplier=2.5)

        # 3. 针对小样本的缺失值填补

        # 对百分比类指标使用线性插值
        for col in percent_cols:
            if col in existing_columns:
                region_data = fill_missing_values(region_data, col, method='interpolate')

        # 对计数类指标使用前向后向填充
        for col in count_cols:
            if col in existing_columns:
                region_data = fill_missing_values(region_data, col, method='ffill_bfill')

        # 对数量类指标使用线性插值
        for col in amount_cols:
            if col in existing_columns:
                region_data = fill_missing_values(region_data, col, method='interpolate')

        # 对财务类指标使用时间加权插值
        for col in finance_cols:
            if col in existing_columns:
                region_data = fill_missing_values(region_data, col, method='time_weighted')

        # 4. 对需要四舍五入的指标进行处理
        for col in round_cols:
            if col in existing_columns:
                # 首先确保列是数值型的
                region_data[col] = convert_to_numeric(region_data[col])
                try:
                    # 对于小样本，我们确保不会将数据四舍五入为0
                    values = region_data[col].dropna()
                    if len(values) > 0 and values.min() < 1 and values.max() < 10:
                        # 如果数据范围很小，保留一位小数
                        region_data[col] = region_data[col].round(1)
                    else:
                        region_data[col] = region_data[col].round(0)

                    # 使用pandas的Int64类型，它可以处理NaN值
                    try:
                        region_data[col] = pd.Series(region_data[col], dtype="Int64")
                    except:
                        # 如果无法转换为Int64（例如有小数），保持为浮点数
                        pass
                except Exception as e:
                    print(f"四舍五入 {col} 时出错: {e}")

        # 5. 检查是否所有数据都已成功填补
        missing_counts = region_data.isna().sum()
        if missing_counts.sum() > 0:
            print(f"⚠️ {region_name} 仍有缺失值:")
            for col in missing_counts[missing_counts > 0].index:
                print(f"  - {col}: {missing_counts[col]} 缺失值")
                # 对于小样本，最后使用均值或中位数填补剩余缺失值
                if region_data[col].notnull().sum() > 0:
                    if col in percent_cols:
                        region_data[col] = region_data[col].fillna(region_data[col].median())
                    else:
                        region_data[col] = region_data[col].fillna(region_data[col].mean())

        print(f'✅ {region_name} 数据处理完成')
        return region_data

    except Exception as e:
        print(f"❌ {region_name} 数据处理出错: {e}")
        return region_data  # 返回原始数据，避免数据丢失


def analyze_sample_size(data):
    """分析每个地区的样本大小和数据质量"""
    print("\n=== 样本大小分析 ===")

    region_counts = data.groupby('地区').size()
    print(f"地区数量: {len(region_counts)}")
    print(f"平均每个地区的样本大小: {region_counts.mean():.1f}")
    print(f"最小样本大小: {region_counts.min()} (地区: {region_counts.idxmin()})")
    print(f"最大样本大小: {region_counts.max()} (地区: {region_counts.idxmax()})")

    missing_data = data.isna().sum().sum()
    total_cells = data.shape[0] * data.shape[1]
    print(f"缺失数据占比: {missing_data / total_cells * 100:.1f}%")

    return region_counts


def main():
    file_path = 'data2.xlsx'
    if not os.path.exists(file_path):
        print(f'❌ 文件 {file_path} 不存在！')
        return

    try:
        data = pd.read_excel(file_path, engine='openpyxl')
        print(f"✅ 成功读取数据，共 {data.shape[0]} 行 {data.shape[1]} 列")

        # 分析样本大小
        region_counts = analyze_sample_size(data)

    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return

    if '年份' not in data.columns or '地区' not in data.columns:
        print("❌ 缺少 '年份' 或 '地区' 列，请检查 Excel 文件格式！")
        return

    all_data = []

    for region in regions:
        print(f"\n🚀 正在处理 {region} 数据...")
        region_data = data[data['地区'] == region]

        if not region_data.empty:
            print(f"  - 样本大小: {len(region_data)}")
            processed_data = process_region_data(region_data.copy(), region)
            all_data.append(processed_data)
        else:
            print(f"⚠️ 没有找到 {region} 的数据")

    if all_data:
        try:
            final_data = pd.concat(all_data, ignore_index=True)

            # 最终质量检查
            missing_cols = final_data.columns[final_data.isna().sum() > 0]
            if len(missing_cols) > 0:
                print("\n⚠️ 最终数据仍有缺失值:")
                for col in missing_cols:
                    missing_count = final_data[col].isna().sum()
                    missing_pct = missing_count / len(final_data) * 100
                    print(f"  - {col}: {missing_count} 缺失值 ({missing_pct:.1f}%)")

            # 保存数据
            final_data.to_excel(output_file, index=False, engine='openpyxl')
            print(f"\n✅ 数据处理完成！已保存至 {output_file}")

            # 输出统计摘要
            print("\n=== 数据处理统计 ===")
            print(f"总行数: {len(final_data)}")
            print(f"总地区数: {final_data['地区'].nunique()}")
            print(f"年份范围: {final_data['年份'].min()} - {final_data['年份'].max()}")

        except Exception as e:
            print(f"❌ 保存数据出错: {e}")
    else:
        print("❌ 没有数据被处理，请检查输入数据")


if __name__ == "__main__":
    # 可以在这里导入scipy的interpolate
    try:
        from scipy.interpolate import interp1d
    except ImportError:
        print("⚠️ 无法导入scipy.interpolate，将使用替代方法进行插值")

    main()