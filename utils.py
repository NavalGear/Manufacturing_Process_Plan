"""
Các tiện ích cho hệ thống Đa Tác Tử AI cho Biên Soạn Quy Trình Công Nghệ Chế Tạo
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import fitz  # PyMuPDF
import config

logger = logging.getLogger(__name__)

# Custom JSON encoder to handle CrewOutput objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle CrewOutput objects
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # Let the base class handle other types
        return super().default(obj)

def load_drawing_file(file_path: str) -> str:
    """
    Tải nội dung từ file bản vẽ kỹ thuật (hỗ trợ text và PDF)
    
    Args:
        file_path: Đường dẫn đến file bản vẽ
        
    Returns:
        Nội dung của file dưới dạng chuỗi
    """
    logger.info(f"Đang tải file bản vẽ từ: {file_path}")
    
    try:
        # Kiểm tra loại file
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            # Xử lý file PDF
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        elif ext in ['.txt', '.md']:
            # Xử lý file văn bản
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Không hỗ trợ định dạng file: {ext}")
            
    except Exception as e:
        logger.error(f"Lỗi khi tải file bản vẽ: {str(e)}")
        raise

def save_data(data: Any, file_name: str, directory: str = config.DATA_PATH) -> str:
    """
    Lưu dữ liệu vào file JSON
    
    Args:
        data: Dữ liệu cần lưu
        file_name: Tên file
        directory: Thư mục lưu trữ
        
    Returns:
        Đường dẫn đến file đã lưu
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(directory, exist_ok=True)
    
    # Thêm timestamp vào tên file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{os.path.splitext(file_name)[0]}_{timestamp}.json"
    file_path = os.path.join(directory, file_name)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # Use the custom encoder to handle CrewOutput objects
            json.dump(data, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        logger.info(f"Đã lưu dữ liệu vào: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Lỗi khi lưu dữ liệu: {str(e)}")
        # Create a simplified version of the data if serialization fails
        try:
            simplified_data = {"error": "Original data could not be serialized", "message": str(data)}
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu dữ liệu đơn giản hóa vào: {file_path}")
            return file_path
        except:
            logger.error("Không thể lưu dữ liệu đơn giản hóa")
            raise

def load_data(file_path: str) -> Any:
    """
    Tải dữ liệu từ file JSON
    
    Args:
        file_path: Đường dẫn đến file
        
    Returns:
        Dữ liệu đã tải
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Đã tải dữ liệu từ: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu: {str(e)}")
        raise

def extract_technical_specs(drawing_text: str) -> Dict[str, Any]:
    """
    Trích xuất thông số kỹ thuật từ văn bản bản vẽ
    (Trong phiên bản MVP, đây là hàm mô phỏng đơn giản)
    
    Args:
        drawing_text: Văn bản của bản vẽ kỹ thuật
        
    Returns:
        Dict chứa các thông số kỹ thuật
    """
    specs = {
        "name": "",
        "part_number": "",
        "revision": "",
        "material": "",
        "dimensions": {},
        "surface_finish": {},
        "tolerances": {},
        "heat_treatment": {},
        "notes": []
    }
    
    # Phân tích cơ bản - trong thực tế sẽ cần AI tinh vi hơn
    lines = drawing_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('=='): 
            continue
            
        # Xác định phần
        if "PART NAME:" in line:
            specs["name"] = line.split("PART NAME:")[1].strip()
        elif "PART NUMBER:" in line:
            specs["part_number"] = line.split("PART NUMBER:")[1].strip()
        elif "REVISION:" in line:
            specs["revision"] = line.split("REVISION:")[1].strip()
        elif "MATERIAL:" in line:
            specs["material"] = line.split("MATERIAL:")[1].strip()
        
        # Xác định section
        elif line == "DIMENSIONS:":
            current_section = "dimensions"
        elif line == "SURFACE FINISH:":
            current_section = "surface_finish"
        elif line == "TOLERANCES:":
            current_section = "tolerances"
        elif line == "HEAT TREATMENT:":
            current_section = "heat_treatment"
        elif line == "NOTES:":
            current_section = "notes"
        elif line == "APPROVAL:":
            current_section = "approval"
            
        # Xử lý mục trong section
        elif current_section and line.startswith("- "):
            item = line[2:].strip()
            
            if current_section == "notes":
                specs["notes"].append(item)
            elif current_section in ["dimensions", "surface_finish", "tolerances", "heat_treatment"]:
                if ":" in item:
                    key, value = item.split(":", 1)
                    specs[current_section][key.strip()] = value.strip()
                else:
                    # Thêm như một mục không có key
                    if "items" not in specs[current_section]:
                        specs[current_section]["items"] = []
                    specs[current_section]["items"].append(item)
    
    return specs

def calculate_machining_time(part_specs: Dict[str, Any]) -> Dict[str, float]:
    """
    Tính toán thời gian gia công dựa trên thông số kỹ thuật
    (Trong phiên bản MVP, đây là hàm mô phỏng đơn giản)
    
    Args:
        part_specs: Thông số kỹ thuật của chi tiết
        
    Returns:
        Dict chứa thời gian gia công cho từng công đoạn
    """
    # Mô phỏng tính toán thời gian gia công
    # Trong thực tế sẽ cần các công thức phức tạp dựa trên
    # vật liệu, kích thước, dung sai, v.v.
    
    times = {
        "setup": 30.0,  # phút
        "turning": 0.0,
        "milling": 0.0,
        "drilling": 0.0,
        "grinding": 0.0,
        "heat_treatment": 0.0,
        "inspection": 0.0,
        "total": 0.0
    }
    
    # Mô phỏng tính toán thời gian tiện
    if part_specs.get("dimensions", {}).get("Overall Length"):
        length_str = part_specs["dimensions"]["Overall Length"]
        # Trích xuất số từ chuỗi, ví dụ "120mm ± 0.05mm" -> 120
        import re
        length_match = re.search(r'(\d+\.?\d*)', length_str)
        if length_match:
            length = float(length_match.group(1))
            times["turning"] = length * 0.5  # Mô phỏng: 0.5 phút/mm
    
    # Mô phỏng tính toán thời gian phay
    if "Keyway Width" in part_specs.get("dimensions", {}):
        times["milling"] = 45.0  # Mô phỏng: 45 phút cho phay then
    
    # Tính toán tổng thời gian
    times["heat_treatment"] = 120.0 if "hardness" in part_specs.get("heat_treatment", {}) else 0.0
    times["inspection"] = 15.0  # Mô phỏng: 15 phút cho kiểm tra
    
    # Tổng thời gian
    times["total"] = sum(v for k, v in times.items() if k != "total")
    
    return times 