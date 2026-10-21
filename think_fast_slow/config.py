import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env from project root or parent directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

@dataclass
class DualProcessConfig:
    # System 1: TypeSafe Jev
    typesafe_api_key: str = field(
        default_factory=lambda: os.getenv("TYPESAFE_API_KEY", "")
    )
    typesafe_base_url: str = field(
        default_factory=lambda: os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1")
    )
    typesafe_default_model: str = field(
        default_factory=lambda: os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-latest")
    )

    # System 2: Local Qwen3-8B
    qwen_api_base: str = field(
        default_factory=lambda: os.getenv("QWEN_API_BASE", "http://127.0.0.1:8088/v1")
    )
    qwen_api_key: str = field(
        default_factory=lambda: os.getenv("QWEN_API_KEY", "none")
    )
    qwen_model: str = field(
        default_factory=lambda: os.getenv("QWEN_MODEL", "qwen3-8b")
    )

    # Local Engine Binary & Model Paths
    llama_server_path: str = field(
        default_factory=lambda: os.getenv(
            "LLAMA_SERVER_PATH",
            r"C:\Users\LENOVO\.docker\bin\inference\llama-server.exe",
        )
    )
    qwen_gguf_path: str = field(
        default_factory=lambda: os.getenv(
            "QWEN_GGUF_PATH",
            r"C:\Users\LENOVO\LLM_ThumbDrive_exfat_backup\models\Qwen3-8B-Q4_K_M.gguf",
        )
    )
    llamafile_path: str = field(
        default_factory=lambda: os.getenv(
            "LLAMAFILE_PATH",
            r"C:\Users\LENOVO\LLM_ThumbDrive_exfat_backup\llamafile.exe",
        )
    )

    # Effort Allocation Hyperparameters (Paper Section 5.4 & Table 3)
    r: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_R", "0.1"))
    )  # Overthinking ratio
    gamma: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_GAMMA", "2.0"))
    )  # Ambiguity exponent
    lambda_beta: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_LAMBDA_BETA", "0.01"))
    )  # Cost-scale product
    beta: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_BETA", "1024"))
    )  # Scale tokens
    epsilon: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_EPSILON", "0.35"))
    )  # Give-up error tolerance

    # Calibration & Conformal Defaults
    alpha: float = 0.05  # Conformal error level (95% target coverage)
    delta: float = 0.05  # Family-wise error rate bound for act band

    @property
    def cost_lambda(self) -> float:
        return self.lambda_beta / self.beta

config = DualProcessConfig()
