import os
import pandas as pd
from 统计建模.process_data import process_region_data, regions, output_file
from 统计建模.predict_data import predict_future_data

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
                predict_data = predict_future_data(processed_data,region,years_to_predict=[2023, 2024])
                all_data.append(predict_data)
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

if __name__ == '__main__':
    main()