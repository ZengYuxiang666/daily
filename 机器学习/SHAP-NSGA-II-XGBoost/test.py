import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1️⃣ 生成模拟数据
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=20, random_state=42)

# 转换为 DataFrame
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2,
                                                    random_state=42)

# 2️⃣ 训练初始 XGBoost 模型（去掉 use_label_encoder=False 避免警告）
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 3️⃣ 计算 SHAP 特征重要性
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_train)

# 计算特征重要性
shap_importance = np.abs(shap_values.values).mean(axis=0)
sorted_indices = np.argsort(shap_importance)[::-1]  # 按重要性降序排列

# 4️⃣ NSGA-II 进行特征选择
POP_SIZE = 50  # 种群大小
NGEN = 20  # 迭代次数
CX_PB = 0.5  # 交叉概率
MUT_PB = 0.2  # 变异概率

# 创建适应度（优化目标：最大化准确率，同时最小化特征数）
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # (最大化准确率, 最小化特征数量)
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 2)  # 0或1，表示是否选择该特征
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 适应度评估函数
def evaluate(individual):
    selected_features = [feature_names[i] for i in range(len(individual)) if individual[i] == 1]
    if len(selected_features) == 0:  # 避免无特征情况
        return 0, len(individual)

    X_train_subset = X_train[selected_features]
    X_test_subset = X_test[selected_features]

    model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train_subset, y_train)
    y_pred = model.predict(X_test_subset)
    acc = accuracy_score(y_test, y_pred)

    return acc, len(selected_features)


# 注册遗传算法操作
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

# 5️⃣ 运行 NSGA-II 进行特征选择
pop = toolbox.population(n=POP_SIZE)
hof = tools.ParetoFront()  # 存储最优个体

algorithms.eaMuPlusLambda(pop, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, cxpb=CX_PB, mutpb=MUT_PB, ngen=NGEN,
                          stats=None, halloffame=hof, verbose=True)

# 获取最佳个体（即最优特征子集）
best_individual = sorted(hof, key=lambda ind: ind.fitness.values[0], reverse=True)[0]  # 选择最高准确率的个体
selected_features = [feature_names[i] for i in range(len(best_individual)) if best_individual[i] == 1]

print(f"选出的最佳特征数: {len(selected_features)}")
print(f"最佳特征子集: {selected_features}")

# 6️⃣ 用优化后的特征重新训练 XGBoost
X_train_opt = X_train[selected_features]
X_test_opt = X_test[selected_features]

opt_xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
opt_xgb_model.fit(X_train_opt, y_train)

y_pred_opt = opt_xgb_model.predict(X_test_opt)
opt_acc = accuracy_score(y_test, y_pred_opt)

print(f"优化后 XGBoost 模型准确率: {opt_acc:.4f}")
