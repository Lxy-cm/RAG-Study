import os
from dotenv import load_dotenv

# 强制加载项目根目录下的 .env 文件
load_dotenv()

class Config:
    """全局配置中心"""

    # ==========================================
    # 1. 安全与鉴权 (绝不硬编码！)
    # ==========================================
    LLM_API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

    # 阿里云百炼 / 通义千问 OpenAI 兼容接口。
    # 如果使用其他地域，可以在 .env 里覆盖 LLM_BASE_URL。
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # ==========================================
    # 2. 路径配置
    # ==========================================
    # 向量数据库的持久化目录
    DB_DIR = os.getenv("DB_DIR", "./database")
    # 原始 Markdown 文档存放目录
    DATA_DIR = os.getenv("DATA_DIR", "./data")

    # ==========================================
    # 3. 模型参数配置
    # ==========================================
    # 大语言模型 (生成回答)
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-plus")
    LLM_TEMPERATURE = 0.1  # 数学问题需要严谨，温度设低

    # 向量化模型 (本地运行)
    EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    
    # 交叉重排模型 (本地运行)
    RERANK_MODEL_NAME = "BAAI/bge-reranker-large"

    # ==========================================
    # 4. 硬件与性能配置
    # ==========================================
    # 本地模型运行设备。默认自动判断；CPU 版 torch 环境下不能写死 cuda。
    @staticmethod
    def _detect_device():
        configured_device = os.getenv("DEVICE")
        if configured_device:
            return configured_device

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    DEVICE = _detect_device()
    
    # 检索相关参数
    RETRIEVER_TOP_K = 10     # 第一阶段向量召回的数量
    RERANK_TOP_K = 2         # 第二阶段精排后喂给大模型的数量
