"""
Ứng dụng chính cho hệ thống Đa Tác Tử AI Biên Soạn Quy Trình Công Nghệ Chế Tạo
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from crewai import Crew, Process
import config
import utils
import agents
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()

# Thiết lập logging
logger = logging.getLogger(__name__)

def parse_arguments():
    """Xử lý tham số dòng lệnh"""
    parser = argparse.ArgumentParser(
        description='Hệ thống Đa Tác Tử AI Biên Soạn Quy Trình Công Nghệ Chế Tạo'
    )
    
    parser.add_argument(
        '--drawing', '-d',
        type=str,
        default='drawings/sample_part.txt',
        help='Đường dẫn tới file bản vẽ kỹ thuật (PDF hoặc TXT)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='manufacturing_plan.json',
        help='Tên file để lưu kết quả'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Hiển thị thông tin chi tiết'
    )
    
    return parser.parse_args()

def create_manufacturing_crew(drawing_path: str) -> Crew:
    """
    Tạo nhóm các tác tử cho quy trình sản xuất
    
    Args:
        drawing_path: Đường dẫn đến file bản vẽ
        
    Returns:
        Crew: Nhóm tác tử đã cấu hình
    """
    # Tạo các tác tử
    design_analyzer = agents.create_design_analyzer_agent()
    material_specialist = agents.create_material_selection_agent()
    tooling_specialist = agents.create_tooling_selection_agent()
    process_planner = agents.create_process_planning_agent()
    quality_controller = agents.create_quality_control_agent()
    orchestrator = agents.create_orchestrator_agent()
    
    # Load drawing content directly to pass as context
    try:
        with open(drawing_path, 'r', encoding='utf-8') as f:
            drawing_content = f.read()
        print(f"Successfully loaded drawing file: {drawing_path}")
        print(f"Drawing content length: {len(drawing_content)} characters")
    except Exception as e:
        print(f"Error loading drawing file: {str(e)}")
        drawing_content = ""
    
    # Tạo các task
    analyze_drawing_task = agents.create_analyze_drawing_task(
        design_analyzer, drawing_path
    )
    
    material_selection_task = agents.create_material_selection_task(
        material_specialist,
        context=[analyze_drawing_task]
    )
    
    tooling_selection_task = agents.create_tooling_selection_task(
        tooling_specialist,
        context=[analyze_drawing_task, material_selection_task]
    )
    
    process_plan_task = agents.create_process_plan_task(
        process_planner, 
        context=[analyze_drawing_task, material_selection_task, tooling_selection_task]
    )
    
    quality_review_task = agents.create_quality_review_task(
        quality_controller,
        context=[process_plan_task, material_selection_task, tooling_selection_task]
    )
    
    finalize_plan_task = agents.create_finalize_plan_task(
        orchestrator,
        context=[process_plan_task, quality_review_task, material_selection_task, tooling_selection_task]
    )
    
    # Tạo crew
    manufacturing_crew = Crew(
        agents=[design_analyzer, material_specialist, tooling_specialist, process_planner, quality_controller, orchestrator],
        tasks=[analyze_drawing_task, material_selection_task, tooling_selection_task, process_plan_task, quality_review_task, finalize_plan_task],
        verbose=config.VERBOSE,
        process=Process.sequential  # Chạy tuần tự
    )
    
    return manufacturing_crew

def run_manufacturing_planning(drawing_path: str) -> Dict[str, Any]:
    """
    Chạy quy trình lập kế hoạch sản xuất
    
    Args:
        drawing_path: Đường dẫn đến file bản vẽ
        
    Returns:
        Dict chứa kế hoạch sản xuất
    """
    logger.info(f"Bắt đầu quy trình lập kế hoạch sản xuất cho bản vẽ: {drawing_path}")
    
    # Kiểm tra file bản vẽ
    if not os.path.exists(drawing_path):
        logger.error(f"Không tìm thấy file bản vẽ: {drawing_path}")
        raise FileNotFoundError(f"Không tìm thấy file bản vẽ: {drawing_path}")
    
    # Tạo nhóm tác tử
    crew = create_manufacturing_crew(drawing_path)
    
    try:
        # Chạy quy trình
        logger.info("Đang chạy các tác tử AI...")
        result = crew.kickoff()
        
        # Tạo cấu trúc dữ liệu kết quả
        manufacturing_plan = {
            "source_drawing": drawing_path,
            "timestamp": utils.datetime.now().isoformat(),
            "plan": result
        }
        
        logger.info("Đã hoàn thành quy trình lập kế hoạch sản xuất")
        return manufacturing_plan
        
    except Exception as e:
        logger.error(f"Lỗi khi chạy quy trình: {str(e)}")
        raise

def main():
    """Hàm chính của ứng dụng"""
    # Xử lý tham số dòng lệnh
    args = parse_arguments()
    
    # Ghi đè cấu hình nếu được chỉ định
    if args.verbose:
        config.VERBOSE = True
    
    try:
        # Chạy quy trình lập kế hoạch sản xuất
        manufacturing_plan = run_manufacturing_planning(args.drawing)
        
        # Lưu kết quả
        output_path = utils.save_data(manufacturing_plan, args.output)
        print(f"\nĐã lưu kế hoạch sản xuất vào: {output_path}")
        
        # Hiển thị tóm tắt
        print("\n===== TÓM TẮT KẾ HOẠCH SẢN XUẤT =====")
        print(f"Bản vẽ nguồn: {args.drawing}")
        print(f"Kết quả:")
        print(manufacturing_plan["plan"])
        
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        print(f"Đã xảy ra lỗi: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 