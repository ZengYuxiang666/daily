import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import shap

# 1. 读取数据集
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/german_credit_data/german_credit_data.csv"
df = pd.read_csv(url)

# 2. 预处理数据
# 删除无关列
df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

# 3. 处理类别变量
df = pd.get_dummies(df, drop_first=True)  # One-hot 编码

# 4. 选择特征和标签
X = df.drop(columns=["Creditability"])  # 预测变量
y = df["Creditability"]  # 目标变量（信用好/差）

# 5. 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. 训练 SVM 模型
svm_model = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 8. 预测
y_pred = svm_model.predict(X_test)

# 9. 评估模型
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 10. 使用 SHAP 解释模型
explainer = shap.Explainer(svm_model.predict_proba, X_train)
shap_values = explainer(X_test)

# 11. 画出 SHAP 特征重要性图
shap.summary_plot(shap_values.values[:, :, 1], X_test, feature_names=X.columns)
