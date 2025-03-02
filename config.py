"""
Cấu hình hệ thống Đa Tác Tử AI cho Biên Soạn Quy Trình Công Nghệ Chế Tạo Cơ Khí
"""

import os
from dotenv import load_dotenv
import logging
import sys

# Tải biến môi trường từ file .env
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Cấu hình hệ thống
VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ensure logs directory exists
logs_path = os.getenv("LOGS_PATH", "./logs")
os.makedirs(logs_path, exist_ok=True)

# Thiết lập logging
logging_level = getattr(logging, LOG_LEVEL)

# Create a custom formatter that handles Unicode characters
class UnicodeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Replace Vietnamese characters that might cause issues in Windows console
            msg = msg.replace('Đ', 'D').replace('đ', 'd')
            msg = msg.replace('ă', 'a').replace('â', 'a').replace('ê', 'e')
            msg = msg.replace('ô', 'o').replace('ơ', 'o').replace('ư', 'u')
            msg = msg.replace('ế', 'e').replace('ề', 'e').replace('ể', 'e')
            msg = msg.replace('ố', 'o').replace('ồ', 'o').replace('ổ', 'o')
            msg = msg.replace('ứ', 'u').replace('ừ', 'u').replace('ử', 'u')
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(logs_path, "manufacturing_agents.log"), encoding='utf-8'),
        UnicodeStreamHandler() if sys.platform == 'win32' else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Đường dẫn dữ liệu
DRAWINGS_PATH = os.getenv("DRAWINGS_PATH", "./drawings")
DATA_PATH = os.getenv("DATA_PATH", "./data")
LOGS_PATH = os.getenv("LOGS_PATH", "./logs")
MODELS_PATH = os.getenv("MODELS_PATH", "./models")

# Cấu hình mô hình
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# Cấu hình tác tử
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
TIMEOUT = 1800  # 30 phút