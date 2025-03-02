"""
Script chạy tác tử AutoGen nâng cao cho hệ thống Đa Tác Tử AI
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import config
import utils
import autogen_agents

# Tải biến môi trường
load_dotenv()

# Thiết lập logging
logger = logging.getLogger(__name__)

def parse_arguments():
    """Xử lý tham số dòng lệnh"""
    parser = argparse.ArgumentParser(
        description='Chạy tác tử AutoGen nâng cao cho hệ thống Đa Tác Tử AI'
    )
    
    parser.add_argument(
        '--drawing', '-d',
        type=str,
        default='drawings/sample_part.txt',
        help='Đường dẫn tới file bản vẽ kỹ thuật (PDF hoặc TXT)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['code', 'team'],
        default='team',
        help='Chế độ chạy: "code" (tác tử thực thi mã) hoặc "team" (nhóm tác tử giao tiếp)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='autogen_result.json',
        help='Tên file để lưu kết quả'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Hiển thị thông tin chi tiết'
    )
    
    return parser.parse_args()

def run_code_execution_agent(drawing_path: str) -> Dict[str, Any]:
    """
    Chạy tác tử AutoGen thực thi mã
    
    Args:
        drawing_path: Đường dẫn đến file bản vẽ
        
    Returns:
        Dict chứa kết quả tối ưu hóa
    """
    logger.info(f"Chạy tác tử AutoGen thực thi mã với bản vẽ: {drawing_path}")
    
    # Kiểm tra file bản vẽ
    if not os.path.exists(drawing_path):
        logger.error(f"Không tìm thấy file bản vẽ: {drawing_path}")
        raise FileNotFoundError(f"Không tìm thấy file bản vẽ: {drawing_path}")
    
    # Tải nội dung bản vẽ
    drawing_text = utils.load_drawing_file(drawing_path)
    
    # Trích xuất thông số kỹ thuật
    part_specs = utils.extract_technical_specs(drawing_text)
    
    # Chạy tác tử thực thi mã
    result = autogen_agents.create_code_execution_agent(part_specs)
    
    return {
        "source_drawing": drawing_path,
        "part_specs": part_specs,
        "optimization_result": result
    }

def run_team_chat(drawing_path: str) -> Dict[str, Any]:
    """
    Chạy nhóm tác tử AutoGen giao tiếp
    
    Args:
        drawing_path: Đường dẫn đến file bản vẽ
        
    Returns:
        Dict chứa kết quả của cuộc thảo luận
    """
    logger.info(f"Chạy nhóm tác tử AutoGen với bản vẽ: {drawing_path}")
    
    # Kiểm tra file bản vẽ
    if not os.path.exists(drawing_path):
        logger.error(f"Không tìm thấy file bản vẽ: {drawing_path}")
        raise FileNotFoundError(f"Không tìm thấy file bản vẽ: {drawing_path}")
    
    # Chạy nhóm tác tử
    result = autogen_agents.create_manufacturing_team_chat(drawing_path)
    
    return {
        "source_drawing": drawing_path,
        "team_chat_result": result
    }

def main():
    """Hàm chính của script"""
    # Xử lý tham số dòng lệnh
    args = parse_arguments()
    
    # Ghi đè cấu hình nếu được chỉ định
    if args.verbose:
        config.VERBOSE = True
    
    try:
        # Chạy tác tử phù hợp với chế độ đã chọn
        if args.mode == 'code':
            result = run_code_execution_agent(args.drawing)
        else:  # mode == 'team'
            result = run_team_chat(args.drawing)
        
        # Lưu kết quả
        output_path = utils.save_data(result, args.output)
        print(f"\nĐã lưu kết quả vào: {output_path}")
        
        # Hiển thị tóm tắt
        print(f"\n===== TÓM TẮT KẾT QUẢ ({args.mode}) =====")
        print(f"Bản vẽ nguồn: {args.drawing}")
        
        if args.mode == 'code':
            if "error" in result.get("optimization_result", {}):
                print(f"Lỗi: {result['optimization_result']['error']}")
            else:
                print("Quá trình tối ưu hóa đã hoàn thành thành công")
        else:  # mode == 'team'
            if "final_plan" in result.get("team_chat_result", {}):
                print("Kế hoạch sản xuất đã được tạo thành công")
            else:
                print("Cuộc thảo luận nhóm đã hoàn thành nhưng không tìm thấy kế hoạch cuối cùng rõ ràng")
        
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        print(f"Đã xảy ra lỗi: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 