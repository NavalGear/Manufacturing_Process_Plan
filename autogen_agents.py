"""
Tác tử nâng cao sử dụng AutoGen cho hệ thống Đa Tác Tử AI cho Biên Soạn Quy Trình Công Nghệ Chế Tạo
"""

import os
import json
import logging
import autogen
from typing import Dict, List, Any, Optional, Union
import config

logger = logging.getLogger(__name__)

def create_code_execution_agent(part_specs: Dict):
    """
    Tạo một tác tử AutoGen có khả năng chạy mã để tính toán tối ưu quy trình sản xuất.
    
    Args:
        part_specs: Thông số kỹ thuật của chi tiết
        
    Returns:
        Kết quả của tính toán
    """
    # Cấu hình cho mô hình LLM
    config_list = [
        {
            "model": config.DEFAULT_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": config.TEMPERATURE
        }
    ]
    
    # Tạo một assistant có khả năng chạy mã
    assistant = autogen.AssistantAgent(
        name="Manufacturing_Optimization_Assistant",
        llm_config={
            "config_list": config_list,
            "temperature": 0.3,
        },
        system_message="""
        You are a manufacturing optimization specialist with expertise in CNC programming and manufacturing process optimization.
        You should use Python code to solve manufacturing optimization problems and perform calculations.
        You must provide well-documented code that includes clear explanations of your approach.
        """
    )
    
    # Tạo một user proxy agent có khả năng thực thi mã
    user_proxy = autogen.UserProxyAgent(
        name="Manufacturing_Engineer",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": "execution_env",
            "use_docker": False  # Sử dụng môi trường Python cục bộ
        },
        system_message="""
        You are a manufacturing engineer who needs to optimize a manufacturing process.
        You will be given a set of specifications and need to calculate optimal machining parameters, 
        estimate machining time, and provide visualization of the process.
        """
    )
    
    # Khởi tạo thư mục làm việc
    os.makedirs("execution_env", exist_ok=True)
    
    # Lưu thông số kỹ thuật vào file JSON cho tác tử sử dụng
    with open("execution_env/part_specs.json", "w", encoding='utf-8') as f:
        json.dump(part_specs, f, ensure_ascii=False, indent=2)
    
    # Tạo prompt cho tính toán
    optimization_prompt = f"""
    Based on the part specifications in the file 'part_specs.json', please:
    
    1. Calculate optimal cutting speeds and feeds for each machining operation based on the material properties
    2. Estimate the machining time for each operation
    3. Generate a simple visualization of the part based on dimensions
    4. Identify potential bottlenecks in the manufacturing process
    5. Suggest ways to optimize the process and reduce production time
    
    Use Python for all calculations and visualizations. Please save all results to a file called 'optimization_results.json'.
    """
    
    try:
        # Khởi động cuộc hội thoại giữa các tác tử
        user_proxy.initiate_chat(
            assistant,
            message=optimization_prompt
        )
        
        # Kiểm tra và tải kết quả
        result_path = "execution_env/optimization_results.json"
        if os.path.exists(result_path):
            with open(result_path, "r", encoding='utf-8') as f:
                results = json.load(f)
            return results
        else:
            logger.error("Không tìm thấy file kết quả tối ưu hóa")
            return {"error": "Không tìm thấy file kết quả tối ưu hóa"}
            
    except Exception as e:
        logger.error(f"Lỗi khi chạy tác tử AutoGen: {str(e)}")
        return {"error": str(e)}

def create_manufacturing_team_chat(drawing_path: str):
    """
    Tạo một nhóm chat giữa các tác tử AutoGen cho quy trình sản xuất.
    
    Args:
        drawing_path: Đường dẫn đến file bản vẽ
        
    Returns:
        Kết quả của cuộc thảo luận
    """
    # Cấu hình cho mô hình LLM
    config_list = [
        {
            "model": config.DEFAULT_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": config.TEMPERATURE
        }
    ]
    
    # Tạo các tác tử trong nhóm
    design_engineer = autogen.AssistantAgent(
        name="Design_Engineer",
        system_message="""
        You are a Design Analysis Engineer who extracts manufacturing specifications from technical drawings.
        Your goal is to identify all critical dimensions, tolerances, material requirements, and other manufacturing 
        specifications needed for production.
        """,
        llm_config={"config_list": config_list}
    )
    
    materials_engineer = autogen.AssistantAgent(
        name="Materials_Engineer",
        system_message="""
        You are a Materials Engineer who specializes in selecting optimal materials for manufacturing processes.
        Your goal is to analyze the design specifications and recommend the best materials based on performance 
        requirements, manufacturability, cost-efficiency, and sustainability factors. You should evaluate the 
        specified materials and suggest alternatives that might improve the final product quality or manufacturing 
        efficiency. Consider material properties, processing requirements, availability, and supply chain factors 
        in your recommendations.
        """,
        llm_config={"config_list": config_list}
    )
    
    tooling_specialist = autogen.AssistantAgent(
        name="Tooling_Specialist",
        system_message="""
        You are a Tooling Specialist with extensive expertise in selecting optimal cutting tools, fixtures, and machining parameters.
        Your goal is to analyze the part geometry and material specifications to recommend the most appropriate 
        tooling solutions for each manufacturing operation. You should specify cutting tools, tool holders, fixtures, 
        and optimal cutting parameters (speeds, feeds, depths of cut) for each operation. Consider part tolerances, 
        surface finish requirements, material properties, production efficiency, and cost factors in your recommendations.
        Provide detailed specifications including tool geometries, materials, coatings, and dimensions where applicable.
        """,
        llm_config={"config_list": config_list}
    )
    
    manufacturing_engineer = autogen.AssistantAgent(
        name="Manufacturing_Engineer",
        system_message="""
        You are a Manufacturing Process Planner who creates detailed production plans based on design specifications.
        Your goal is to determine the optimal sequence of operations, select appropriate machining methods,
        and identify the tools and equipment needed for each step.
        """,
        llm_config={"config_list": config_list}
    )
    
    quality_engineer = autogen.AssistantAgent(
        name="Quality_Engineer",
        system_message="""
        You are a Quality Assurance Engineer who reviews manufacturing plans to ensure quality standards.
        Your goal is to identify critical quality checkpoints, suggest inspection methods, and ensure the 
        manufacturing process will meet all quality requirements and specifications.
        """,
        llm_config={"config_list": config_list}
    )
    
    project_manager = autogen.AssistantAgent(
        name="Project_Manager",
        system_message="""
        You are a Manufacturing Project Manager who coordinates the overall process and finalizes plans.
        Your goal is to synthesize input from all team members, optimize for efficiency and cost,
        and create a final manufacturing plan that is comprehensive and ready for implementation.
        You should also manage the conversation to ensure all necessary information is collected and 
        a complete plan is produced.
        """,
        llm_config={"config_list": config_list}
    )
    
    user_proxy = autogen.UserProxyAgent(
        name="Factory_Owner",
        human_input_mode="NEVER",
        system_message="""
        You are the factory owner who needs a comprehensive manufacturing plan for a new part.
        You have provided a technical drawing and need the team to create a detailed plan for 
        production that optimizes quality, efficiency, and cost.
        """
    )
    
    # Tải nội dung file bản vẽ
    try:
        from utils import load_drawing_file
        drawing_content = load_drawing_file(drawing_path)
    except Exception as e:
        logger.error(f"Lỗi khi tải file bản vẽ: {str(e)}")
        drawing_content = f"[Không thể tải file. Lỗi: {str(e)}]"
    
    # Thiết lập nhóm chat
    groupchat = autogen.GroupChat(
        agents=[user_proxy, design_engineer, materials_engineer, tooling_specialist, manufacturing_engineer, quality_engineer, project_manager],
        messages=[],
        max_round=12
    )
    
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})
    
    # Bắt đầu cuộc trò chuyện
    user_proxy.initiate_chat(
        manager,
        message=f"""
        We need to create a manufacturing plan for a new mechanical part.
        Here is the technical drawing specification:
        
        {drawing_content}
        
        Please work together to create a comprehensive manufacturing plan that includes:
        1. Analysis of the technical specifications
        2. Material selection and alternatives analysis
        3. Tooling selection and cutting parameter optimization
        4. Detailed manufacturing process sequence
        5. Quality control checkpoints and methods
        6. Timeline and resource requirements
        
        For the material selection, I would like recommendations on:
        - Whether the specified material is optimal for this application
        - Alternative materials that might improve performance, manufacturability, or cost-efficiency
        - Considerations for sustainability and supply chain reliability
        
        For the tooling selection, I would like recommendations on:
        - Specific cutting tools for each operation (with specifications)
        - Fixtures and workholding solutions
        - Optimal cutting parameters (speeds, feeds, depths of cut)
        - Special tooling considerations for complex features
        
        Let's approach this systematically to create a complete plan ready for implementation.
        """
    )
    
    # Trả về kết quả cuối cùng từ Project Manager
    for message in reversed(groupchat.messages):
        if message.get("sender") == project_manager.name and "final plan" in message.get("content", "").lower():
            return {"final_plan": message.get("content")}
    
    # Nếu không tìm thấy kết quả cuối cùng rõ ràng, trả về toàn bộ cuộc trò chuyện
    return {
        "conversation": [
            {"sender": msg.get("sender"), "content": msg.get("content")} 
            for msg in groupchat.messages
        ]
    } 