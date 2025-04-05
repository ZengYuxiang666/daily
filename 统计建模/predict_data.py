import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')

# 适用于 线性回归（趋势较稳定）
linear_regression_arr = [
    '农作物总播种面积(千公顷)', '耕地灌溉面积（千公顷）', '第一产业/GDP（%）',
    '文化站个数（个）', '移动电话年末用户(万户)', '企业拥有网站数(个)', 'R&D人员全时当量（人）',
    'R&D经费支出(万元)'
]

# 适用于 ARIMA/SARIMA（时间序列分析）
arima_sarima_arr = [
    '粮食总产量（万吨）', '农用化肥施用量（万吨）', '农药使用量（吨）',
    '农村用电量（万千瓦时）', 'GDP增长率（%）', '柴油使用量（万吨）'
]

# 适用于 指数平滑（ETS）和 Prophet（短期趋势预测）
ets_prophet_arr = [
    '农村居民恩格尔系数（%）', '文教娱\n乐支出（%）',
    '第一产业/GDP（%）', '休闲农业与乡村旅游示范县(个)'
]

# 适用于 KNN 回归（非线性关系但数据量少）
knn_regression_arr = [
    '淘宝村数量（个）', '中国美丽休闲乡村数量', '旅行社数', '发明专利申请受理量(项)'
]

# 适用于 贝叶斯回归（小样本且数据不确定性高）
bayesian_ridge_arr = [
    '农林牧渔业总产值（亿元）', '农村居民家庭\n人均纯收入（元/人）',
    'R&D人员全时当量（人）', 'R&D经费支出(万元)', 'GDP增长率（%）'
]


def predict_future_data(processed_data, region, years_to_predict):
    """
    基于不同类型的数据使用合适的预测方法预测未来数据

    参数:
    processed_data (DataFrame): 包含历史数据的DataFrame
    region (str): 需要预测的地区名称
    years_to_predict (list): 需要预测的年份列表

    返回:
    DataFrame: 包含历史数据和预测数据的DataFrame
    """
    # 筛选出特定地区的数据
    region_data = processed_data[processed_data['地区'] == region].copy()

    if region_data.empty:
        print(f"错误: 未找到地区 '{region}' 的数据")
        return None

    # 按年份排序
    region_data = region_data.sort_values('年份')

    # 获取最后一年的数据作为基准
    last_year = region_data['年份'].max()
    last_year_data = region_data[region_data['年份'] == last_year].iloc[0].to_dict()

    # 创建预测数据框架
    future_rows = []
    for year in years_to_predict:
        new_row = last_year_data.copy()
        new_row['年份'] = year
        future_rows.append(new_row)

    forecast_df = pd.DataFrame(future_rows)

    # 数值列（排除年份和地区）
    numeric_cols = [col for col in region_data.columns
                    if col not in ['年份', '地区']
                    and pd.api.types.is_numeric_dtype(region_data[col])]

    # 定义百分比列，确保它们在合理范围内
    percentage_cols = [col for col in numeric_cols if '（%）' in col or '(%)' in col]

    # 对每列进行预测
    for col in numeric_cols:
        # 获取该列的历史数据（过滤掉NaN值）
        data_series = region_data[col].dropna()
        years_series = region_data.loc[data_series.index, '年份']

        # 如果没有足够的数据，跳过预测
        if len(data_series) <= 1:
            print(f"列 '{col}' 数据点不足，使用最后已知值")
            last_valid = data_series.iloc[-1] if not data_series.empty else np.nan
            forecast_df[col] = last_valid
            continue

        # 转换为数组以便建模
        y = data_series.values
        x = years_series.values.reshape(-1, 1)

        # 检查数据量
        data_length = len(y)

        try:
            # 根据列名确定使用的预测方法
            if col in linear_regression_arr:
                # 线性回归适用于趋势稳定的数据
                model = LinearRegression()
                model.fit(x, y)

                for i, year in enumerate(years_to_predict):
                    forecast_df.loc[i, col] = model.predict([[year]])[0]

            elif col in arima_sarima_arr and data_length >= 3:
                # ARIMA模型适用于时间序列数据
                try:
                    # 简单ARIMA模型，参数(p,d,q)=(1,1,0)表示一阶自回归，一阶差分，无移动平均
                    model = ARIMA(y, order=(1, 1, 0))
                    model_fit = model.fit()

                    # 预测未来年份
                    forecast_steps = len(years_to_predict)
                    forecast_values = model_fit.forecast(steps=forecast_steps)

                    for i, pred_value in enumerate(forecast_values):
                        forecast_df.loc[i, col] = pred_value
                except:
                    # 如果ARIMA失败，回退到线性回归
                    print(f"ARIMA预测 '{col}' 失败，回退到线性回归")
                    model = LinearRegression()
                    model.fit(x, y)

                    for i, year in enumerate(years_to_predict):
                        forecast_df.loc[i, col] = model.predict([[year]])[0]

            elif col in ets_prophet_arr and data_length >= 3:
                # 指数平滑适用于短期趋势预测
                try:
                    # 使用Holt-Winters指数平滑
                    # trend可以是'add'或'mul'，seasonal可以是'add'、'mul'或None
                    model = ExponentialSmoothing(y, trend='add', seasonal=None)
                    model_fit = model.fit()

                    # 预测未来年份
                    forecast_steps = len(years_to_predict)
                    forecast_values = model_fit.forecast(forecast_steps)

                    for i, pred_value in enumerate(forecast_values):
                        forecast_df.loc[i, col] = pred_value
                except:
                    # 如果指数平滑失败，回退到线性回归
                    print(f"指数平滑预测 '{col}' 失败，回退到线性回归")
                    model = LinearRegression()
                    model.fit(x, y)

                    for i, year in enumerate(years_to_predict):
                        forecast_df.loc[i, col] = model.predict([[year]])[0]

            elif col in knn_regression_arr and data_length >= 3:
                # KNN回归适用于非线性关系但数据量少的情况
                try:
                    # 选择邻居数量，通常为sqrt(n)或接近的值
                    n_neighbors = min(5, max(2, int(np.sqrt(data_length))))
                    model = KNeighborsRegressor(n_neighbors=n_neighbors)
                    model.fit(x, y)

                    for i, year in enumerate(years_to_predict):
                        forecast_df.loc[i, col] = model.predict([[year]])[0]
                except:
                    # 如果KNN失败，回退到线性回归
                    print(f"KNN回归预测 '{col}' 失败，回退到线性回归")
                    model = LinearRegression()
                    model.fit(x, y)

                    for i, year in enumerate(years_to_predict):
                        forecast_df.loc[i, col] = model.predict([[year]])[0]

            elif col in bayesian_ridge_arr:
                # 贝叶斯回归适用于小样本且数据不确定性高的情况
                try:
                    model = BayesianRidge(n_iter=300, alpha_1=1e-6, alpha_2=1e-6)
                    model.fit(x, y)

                    for i, year in enumerate(years_to_predict):
                        forecast_df.loc[i, col] = model.predict([[year]])[0]
                except:
                    # 如果贝叶斯回归失败，回退到岭回归（带正则化的线性回归）
                    print(f"贝叶斯回归预测 '{col}' 失败，回退到岭回归")
                    model = Ridge(alpha=1.0)
                    model.fit(x, y)

                    for i, year in enumerate(years_to_predict):
                        forecast_df.loc[i, col] = model.predict([[year]])[0]

            else:
                # 对于未分类的列，使用简单线性回归
                model = LinearRegression()
                model.fit(x, y)

                for i, year in enumerate(years_to_predict):
                    forecast_df.loc[i, col] = model.predict([[year]])[0]

            # 应用约束: 确保所有预测值符合实际情况

            # 1. 百分比值约束（非GDP增长率）
            if col in percentage_cols and col != 'GDP增长率（%）':
                forecast_df[col] = forecast_df[col].apply(lambda x: min(100, max(0, x)))

            # 2. GDP增长率特殊约束（一般在-10%到15%之间）
            elif col == 'GDP增长率（%）':
                forecast_df[col] = forecast_df[col].apply(lambda x: min(15, max(-10, x)))

            # 3. 保证非负值（对于不是百分比且不应为负数的列）
            elif col not in percentage_cols:
                forecast_df[col] = forecast_df[col].apply(lambda x: max(0, x))

            # 4. 整数约束（对于本质上是整数的指标）
            integer_cols = ['文化站个数（个）', '旅行社数', '休闲农业与乡村旅游示范县(个)',
                            '中国美丽休闲乡村数量', '淘宝村数量（个）', '企业拥有网站数(个)',
                            '发明专利申请受理量(项)']

            if col in integer_cols:
                forecast_df[col] = forecast_df[col].round().astype('int')

        except Exception as e:
            print(f"预测列 '{col}' 时出错: {e}")
            # 发生错误时，使用最后一个可用值
            if len(y) > 0:
                forecast_df[col] = y[-1]

    # 处理派生指标 (如果需要)
    derive_cols = ['粮食总产量（万吨）/农作物总播种面积(千公顷)', '农用化肥施用量（万吨）/农作物总播种面积(千公顷)',
                   '农药使用量（吨）/农作物总播种面积(千公顷)', 'R&D人员全时当量（人）/R&D经费支出(万元)']

    for col in derive_cols:
        if col in forecast_df.columns:
            # 检查是否需要重新计算
            if '/农作物总播种面积(千公顷)' in col and '农作物总播种面积(千公顷)' in forecast_df.columns:
                numerator = col.split('/')[0]
                if numerator in forecast_df.columns:
                    # 防止除以零
                    denominator = forecast_df['农作物总播种面积(千公顷)'].replace(0, np.nan)
                    forecast_df[col] = forecast_df[numerator] / denominator

            elif col == 'R&D人员全时当量（人）/R&D经费支出(万元)' and 'R&D人员全时当量（人）' in forecast_df.columns and 'R&D经费支出(万元)' in forecast_df.columns:
                # 防止除以零
                denominator = forecast_df['R&D经费支出(万元)'].replace(0, np.nan)
                forecast_df[col] = forecast_df['R&D人员全时当量（人）'] / denominator

    # 处理NaN值
    # 对于预测中的NaN值，使用前一年的值或最近的有效值
    for col in numeric_cols:
        if forecast_df[col].isna().any():
            # 首先尝试用最后一个历史数据替换
            last_valid = region_data[col].dropna().iloc[-1] if not region_data[col].dropna().empty else None
            if last_valid is not None:
                forecast_df[col] = forecast_df[col].fillna(last_valid)

    # 合并历史数据和预测数据
    result_df = pd.concat([region_data, forecast_df], ignore_index=True)

    print(f"地区 {region} 的 {', '.join(map(str, years_to_predict))} 年数据预测完成")
    return result_df