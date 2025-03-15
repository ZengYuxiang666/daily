import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


# 1. 加载乳腺癌威斯康星州数据集
print("加载乳腺癌威斯康星州数据集")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='Class')

# 2. 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("数据归一化")

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("划分数据集和测试集")

# 4. 训练 SVM 模型
print("开始训练SVM模型")
svm_model = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)  # RBF 核适用于非线性分类
svm_model.fit(X_train, y_train)
print("SVM模型训练完成")

# 5. 预测
y_pred = svm_model.predict(X_test)

# 6. 评估模型
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. 使用 SHAP 解释模型
print("使用SHAP解释模型")
explainer = shap.Explainer(svm_model.predict_proba, X_train, n_jobs=-1)  # 采用多核

print("计算SHAP的值（仅使用部分数据加速计算）")
X_test_subset = X_test
shap_values = explainer(X_test_subset)

# 8. 画出 SHAP 特征重要性图
print("画出SHAP特征重要性图")
shap.summary_plot(shap_values.values[:, :, 1], X_test_subset, feature_names=X.columns)

# 9. 使用 PCA 或 t-SNE 降维，绘制 SVM 分类效果图
print("降维并绘制分类效果图")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 重新训练 SVM，确保在降维后的数据上也能分类
svm_model_pca = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
svm_model_pca.fit(X_train_pca, y_train)
y_pred_pca = svm_model_pca.predict(X_test_pca)

# 绘制分类效果图
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, style=y_pred_pca, palette="coolwarm", s=100)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SVM Classification Visualization (PCA)")
plt.legend(title="True Labels & Predictions")
plt.show()
