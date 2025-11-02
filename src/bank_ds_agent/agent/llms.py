import os
import gc
from dotenv import load_dotenv
from llama_cpp import Llama
from langchain_google_genai import ChatGoogleGenerativeAI
from ..tools.code_tool import PythonCode  # 确保导入我们的工具定义

# --- 全局设置 ---
# 加载 .env 文件 (它会读取 LLM_BACKEND, GOOGLE_API_KEY 等)
load_dotenv()
_llm_instance = None  # 缓存

# --- 本地模型路径 (请再次确认) ---
MODEL_PATH_EXECUTOR = "C:/Users/User/.lmstudio/models/Leapps/DeepAnalyze-8B-Q8_0-GGUF/deepanalyze-8b-q8_0.gguf"
GPU_LAYERS_EXECUTOR = -1  # -1 = 全部 VRAM


def get_llm():
    """
    一个“LLM 工厂”函数。
    它会检查 .env 文件中的 LLM_BACKEND 变量，
    然后返回并缓存正确的 LLM 实例 (本地或 API)。
    """
    global _llm_instance
    if _llm_instance:
        return _llm_instance

    backend = os.getenv("LLM_BACKEND", "local")  # 默认为 "local"

    if backend == "api":
        print("--- [LLM 引擎] 正在初始化 Google Gemini API ('api' 模式) ---")
        if not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError(
                "LLM_BACKEND='api'，但 GOOGLE_API_KEY 未在 .env 文件中找到。"
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0,
            convert_system_message_to_human=True,
        )
        # (关键) 将工具绑定到 API LLM
        _llm_instance = llm.bind_tools([PythonCode])
        print("--- [LLM 引擎] Google Gemini API 已准备就绪 (已绑定工具)。 ---")
        return _llm_instance

    elif backend == "local":
        print(f"--- [LLM 引擎] 正在加载 'DeepAnalyze-8B' ('local' 模式) ---")
        llm = Llama(
            model_path=MODEL_PATH_EXECUTOR,
            n_gpu_layers=GPU_LAYERS_EXECUTOR,
            n_ctx=4096,
            verbose=False,
        )
        # (注意: llama-cpp-python 不支持 .bind_tools()。
        #  我们必须依赖 code_generator 的提示词来强制它使用工具格式)
        _llm_instance = llm
        print("--- [LLM 引擎] 本地 8B 模型已加载。 ---")
        return _llm_instance

    else:
        raise ValueError(
            f"未知的 LLM_BACKEND: '{backend}'。请在 .env 中设置为 'local' 或 'api'。"
        )


def unload_llms():
    """(此函数现在仅在本地模式下有用)"""
    global _llm_instance
    _llm_instance = None
    print("--- [LLM 引擎] 正在卸载模型... ---")
    gc.collect()
