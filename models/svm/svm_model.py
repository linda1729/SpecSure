# models/svm/svm_model.py
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class SVMModel:
    def __init__(self, kernel='rbf', C=10, gamma='scale'):
        """初始化 SVM 模型参数"""
        self.clf = SVC(kernel=kernel, C=C, gamma=gamma)

    def train(self, X, y):
        """
        训练模型
        X: 高光谱数据的特征（H x W x B）
        y: 对应的标签（H x W）
        """
        X_flat = X.reshape(-1, X.shape[2])  # 将数据展平，(H*W, B)
        y_flat = y.reshape(-1)  # 展平标签数据

        # 只选择有标签的数据
        valid_idx = y_flat > 0
        X_train = X_flat[valid_idx]
        y_train = y_flat[valid_idx]

        # 训练 SVM 模型
        self.clf.fit(X_train, y_train)
        print("模型训练完成！")

    def predict(self, X):
        """
        预测函数
        X: 高光谱数据的特征（H x W x B）
        """
        X_flat = X.reshape(-1, X.shape[2])  # 展平数据
        y_pred = self.clf.predict(X_flat)  # 做出预测
        return y_pred.reshape(X.shape[0], X.shape[1])  # 重塑为 H x W 形状

    def save(self, model_path):
        """
        保存训练好的模型
        """
        joblib.dump(self.clf, model_path)
        print(f"模型已保存到 {model_path}")

    @staticmethod
    def load(model_path):
        """
        加载已经保存的模型
        """
        clf = joblib.load(model_path)
        model = SVMModel()
        model.clf = clf
        print(f"模型已从 {model_path} 加载")
        return model
