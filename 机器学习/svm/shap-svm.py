import shap
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

"""
SHAP Summary Plot 解读核心
特征排名（Y 轴）：越靠上越重要。
SHAP 影响大小（X 轴）：越远离 0 影响越大。
颜色（特征值大小）：红色（大值），蓝色（小值）。
点的分布：更宽的分布表示该特征对不同样本影响更大。"""

# 生成数据集
X, y = make_classification(n_samples=200, n_features=4, n_informative=2, n_redundant=1, random_state=42)
feature_names = [f'Feature {i}' for i in range(X.shape[1])]

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练 SVM 模型
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 使用 SHAP 进行解释
explainer = shap.Explainer(svm_model.predict_proba, X_train)
shap_values = explainer(X_test)

# 画出 SHAP 特征重要性
shap.summary_plot(shap_values.values[:, :, 1], X_test, feature_names=feature_names)
