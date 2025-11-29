# models/svm/classify.py
from fastapi import APIRouter
from pydantic import BaseModel
from models.svm.svm_model import SVMModel  # 从当前目录下导入 svm_model
from scipy.io import loadmat
import numpy as np

router = APIRouter(prefix="/api/modelA")

class TrainRequest(BaseModel):
    dataset_id: str
    label_path: str
    kernel: str = 'rbf'
    C: float = 10.0
    gamma: str = 'scale'

class PredictRequest(BaseModel):
    dataset_id: str
    model_path: str

# 训练接口
@router.post("/train")
def train_model(req: TrainRequest):
    # 读取数据
    cube = loadmat(f"data/{req.dataset_id}_HHSI.mat")['LN01_HHSI']  # 读取高光谱数据
    label = loadmat(f"data/{req.label_path}")['LN01_GT']  # 读取标签数据

    # 创建 SVM 模型并训练
    model = SVMModel(kernel=req.kernel, C=req.C, gamma=req.gamma)
    model.train(cube, label)

    # 保存模型
    model.save(f"models/svm/{req.dataset_id}_svm_model.joblib")

    return {"status": "ok", "message": f"Model trained and saved as {req.dataset_id}_svm_model.joblib"}

# 预测接口
@router.post("/predict")
def predict(req: PredictRequest):
    # 加载训练好的模型
    model = SVMModel.load(req.model_path)

    # 读取待预测的数据
    cube = loadmat(f"data/{req.dataset_id}_HHSI.mat")['LN01_HHSI']

    # 使用模型进行预测
    predictions = model.predict(cube)

    # 保存预测结果
    np.save(f"data/results/{req.dataset_id}_predictions.npy", predictions)

    return {"status": "ok", "predictions": f"Predictions saved to data/results/{req.dataset_id}_predictions.npy"}
