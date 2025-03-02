"""
Các tác tử AI cho hệ thống Biên Soạn Quy Trình Công Nghệ Chế Tạo Cơ Khí
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool
import config
import utils

logger = logging.getLogger(__name__)

# Khởi tạo tools phổ biến
file_tool = FileReadTool()

# Khởi tạo mô hình LLM
def get_llm(model_name: str = None, temperature: float = None):
    """Tạo một mô hình LLM với các tham số cụ thể"""
    model = model_name or config.DEFAULT_MODEL
    temp = temperature if temperature is not None else config.TEMPERATURE
    
    return ChatOpenAI(
        openai_api_key=config.OPENAI_API_KEY,
        model=model,
        temperature=temp
    )

# 1. Tác tử Phân Tích Bản Vẽ (Design Analyzer Agent)
def create_design_analyzer_agent():
    """Tạo tác tử phân tích bản vẽ kỹ thuật"""
    
    return Agent(
        role="Design Analysis Engineer",
        goal="Analyze technical drawings to extract all relevant manufacturing specifications",
        backstory="""You are an expert mechanical engineer specialized in CAD analysis and design interpretation.
        You can look at technical drawings and extract precise measurements, tolerances, material requirements, 
        and other manufacturing specifications. Your analysis forms the foundation for the entire 
        manufacturing process, so accuracy and completeness are critical.
        
        IMPORTANT: Do NOT attempt to access any external files using file paths. The drawing content 
        will be directly provided to you in your task description. Only use the information directly 
        given to you in the task description between the triple backticks.""",
        verbose=config.VERBOSE,
        allow_delegation=False,
        tools=[file_tool],
        llm=get_llm()
    )

# 1.5 Tác tử Lựa Chọn Vật Liệu Tối Ưu (Material Selection Specialist)
def create_material_selection_agent():
    """Tạo tác tử lựa chọn vật liệu tối ưu"""
    
    return Agent(
        role="Materials Engineer",
        goal="Select optimal materials for manufacturing based on design requirements, performance, cost, and manufacturability",
        backstory="""You are a highly skilled materials engineer with extensive knowledge of engineering 
        materials, their properties, and processing requirements. You understand the complex relationships 
        between material properties, manufacturing processes, costs, and sustainability factors.
        You can analyze design specifications to recommend the most suitable materials, suggest viable 
        alternatives, and optimize material selection to balance performance, manufacturability, cost, 
        and environmental considerations. Your expertise ensures that material choices will meet all 
        technical requirements while optimizing production efficiency and product quality.
        
        IMPORTANT FILE ACCESS RESTRICTIONS:
        
        1. DO NOT attempt to access ANY files EXCEPT for the specific part catalog at:
           "materials_catalogs/SC-2023-A001.txt"
           
        2. DO NOT attempt to access these files (they DO NOT exist):
           - "materials_catalogs/materials_summary.txt"
           - "materials_catalogs/AISI_4140.txt"
           - "materials_catalogs/alloy_steel.txt"
           - "/path/to/document.txt"
           - Any other file not explicitly listed in point #1
        
        3. If the specific catalog file "materials_catalogs/SC-2023-A001.txt" is not found, 
           DO NOT try other files - instead, rely solely on your internal knowledge and 
           the information provided in the context from previous agents.
        
        4. Use ONLY the information provided directly in your context from previous agents and 
           the single authorized catalog file.
        
        5. If you need information that is not available in the context or the authorized catalog file,
           make reasonable assumptions based on standard material properties rather than trying to
           access other unauthorized files.""",
        verbose=config.VERBOSE,
        allow_delegation=False,
        tools=[file_tool],
        llm=get_llm()
    )

# 1.8 Tác tử Lựa Chọn Dụng Cụ Gia Công (Tooling Selection Specialist)
def create_tooling_selection_agent():
    """Tạo tác tử lựa chọn dụng cụ gia công"""
    
    return Agent(
        role="Tooling Specialist",
        goal="Select optimal tooling and fixtures for manufacturing operations based on part geometry, material, and precision requirements",
        backstory="""You are an expert tooling engineer with decades of experience in selecting and 
        optimizing cutting tools, fixtures, and machining parameters. You have deep knowledge of tool 
        geometries, materials, coatings, and their applications across various manufacturing processes 
        including turning, milling, drilling, grinding, and EDM. You understand how tool selection affects 
        surface finish, dimensional accuracy, cycle time, and production costs. Your expertise enables you 
        to recommend the most appropriate tooling solutions for specific manufacturing challenges, 
        considering factors such as workpiece material properties, geometric complexity, tolerance requirements, 
        and production volume. You stay current with the latest tooling technologies and can suggest innovative 
        solutions to improve manufacturing efficiency and quality.
        
        IMPORTANT FILE ACCESS RESTRICTIONS:
        
        1. DO NOT attempt to access ANY files EXCEPT for the specific part catalog at:
           "tooling_catalogs/SC-2023-A001.txt"
           
        2. DO NOT attempt to access these files (they DO NOT exist):
           - "tooling_catalogs/tooling_summary.txt"
           - "tooling_catalogs/cutting_tools.txt"
           - "tooling_catalogs/fixtures.txt"
           - "/path/to/document.txt"
           - Any other file not explicitly listed in point #1
        
        3. If the specific catalog file "tooling_catalogs/SC-2023-A001.txt" is not found, 
           DO NOT try other files - instead, rely solely on your internal knowledge and 
           the information provided in the context from previous agents.
        
        4. Use ONLY the information provided directly in your context from previous agents and 
           the single authorized catalog file.
        
        5. If you need information that is not available in the context or the authorized catalog file,
           make reasonable assumptions based on standard tooling knowledge rather than trying to
           access other unauthorized files.""",
        verbose=config.VERBOSE,
        allow_delegation=False,
        tools=[file_tool],
        llm=get_llm()
    )

# 2. Tác tử Lập Quy Trình Gia Công (Process Planning Agent)
def create_process_planning_agent():
    """Tạo tác tử lập quy trình gia công"""
    
    return Agent(
        role="Manufacturing Process Planner",
        goal="Create detailed manufacturing process plans based on design specifications",
        backstory="""You are a manufacturing process expert with decades of experience.
        You understand machining processes, tooling requirements, and can optimize production 
        for efficiency, cost-effectiveness, and quality. Your expertise covers turning, milling,
        drilling, grinding, and other common manufacturing processes. You can determine the optimal
        sequence of operations to create a part according to specifications.
        
        IMPORTANT FILE ACCESS RESTRICTIONS:
        
        1. DO NOT attempt to access ANY files EXCEPT for the specific part standards at:
           "process_standards/SC-2023-A001.txt"
           
        2. DO NOT attempt to access these files (they DO NOT exist):
           - "tooling_recommendations.txt" (with or without leading slash)
           - "detailed_tooling_recommendations.txt" (with or without leading slash)
           - "process_standards/general_machining.txt"
           - "process_standards/cutting_parameters.txt"
           - "/path/to/document.txt"
           - Any other file not explicitly listed in point #1
        
        3. If the specific standards file "process_standards/SC-2023-A001.txt" is not found, 
           DO NOT try other files - instead, rely solely on your internal knowledge and 
           the information provided in the context from previous agents.
        
        4. Use ONLY the information provided directly in your context from previous agents and 
           the single authorized standards file.
        
        5. If you need information that is not available in the context or the authorized standards file,
           make reasonable assumptions based on standard manufacturing practices rather than trying to
           access other unauthorized files.""",
        verbose=config.VERBOSE,
        allow_delegation=False,
        tools=[file_tool],
        llm=get_llm()
    )

# 3. Tác tử Kiểm Tra Chất Lượng (Quality Control Agent)
def create_quality_control_agent():
    """Tạo tác tử kiểm tra chất lượng"""
    
    return Agent(
        role="Quality Assurance Engineer",
        goal="Evaluate manufacturing plans to ensure they meet quality standards and specifications",
        backstory="""You are a meticulous quality assurance engineer with expertise in 
        mechanical manufacturing. You can detect potential issues in manufacturing plans,
        suggest quality control checkpoints, and ensure the final product will meet all specifications.
        You understand measurement techniques, inspection equipment, and quality standards
        such as ISO 9001.
        
        IMPORTANT FILE ACCESS RESTRICTIONS:
        
        1. DO NOT attempt to access ANY files EXCEPT for the specific quality standards at:
           "quality_standards/SC-2023-A001.txt"
           
        2. DO NOT attempt to access these files (they DO NOT exist):
           - "quality_standards/general_standards.txt"
           - "quality_standards/inspection_methods.txt"
           - "quality_standards/acceptance_criteria.txt"
           - "/path/to/document.txt"
           - Any other file not explicitly listed in point #1
        
        3. If the specific standards file "quality_standards/SC-2023-A001.txt" is not found, 
           DO NOT try other files - instead, rely solely on your internal knowledge and 
           the information provided in the context from previous agents.
        
        4. Use ONLY the information provided directly in your context from previous agents and 
           the single authorized standards file.
        
        5. If you need information that is not available in the context or the authorized standards file,
           make reasonable assumptions based on standard quality practices rather than trying to
           access other unauthorized files.""",
        verbose=config.VERBOSE,
        allow_delegation=False,
        tools=[file_tool],
        llm=get_llm()
    )

# 4. Tác tử Điều Phối (Orchestrator Agent)
def create_orchestrator_agent():
    """Tạo tác tử điều phối tổng thể"""
    
    return Agent(
        role="Manufacturing Project Manager",
        goal="Coordinate the overall manufacturing process planning and optimize for quality and efficiency",
        backstory="""You are a seasoned manufacturing project manager who excels at
        coordinating cross-functional teams. You understand both the technical and business
        aspects of manufacturing, and can synthesize inputs from different specialists to
        create cohesive manufacturing plans. You seek to optimize processes for efficiency,
        quality, and cost-effectiveness.""",
        verbose=config.VERBOSE,
        allow_delegation=True,
        tools=[],
        llm=get_llm(temperature=0.4)  # Nhiệt độ cao hơn cho sự sáng tạo
    )

# Định nghĩa các Task cho từng Agent
def create_analyze_drawing_task(agent, drawing_path: str):
    """Tạo task cho việc phân tích bản vẽ kỹ thuật"""
    
    # Load the drawing content directly
    try:
        with open(drawing_path, 'r', encoding='utf-8') as f:
            drawing_content = f.read()
        print(f"Successfully loaded drawing file in create_analyze_drawing_task: {drawing_path}")
        print(f"Drawing content length: {len(drawing_content)} characters")
    except Exception as e:
        print(f"Error loading drawing file in create_analyze_drawing_task: {str(e)}")
        drawing_content = f"[Error loading drawing file: {str(e)}]"
    
    return Task(
        description=f"""
        IMPORTANT: DO NOT ATTEMPT TO ACCESS ANY FILES. All the information you need is already provided below.
        
        Analyze the following technical drawing specifications that are provided directly here:
        
        ```
        {drawing_content}
        ```
        
        Extract all relevant manufacturing data including:
        1. Dimensions and tolerances
        2. Material specifications
        3. Surface finish requirements
        4. Special features or critical dimensions
        5. Assembly requirements if applicable
        
        Compile your findings in a structured format that can be used by the Process Planner.
        Do NOT attempt to read any external files - use ONLY the content provided above between the triple backticks.
        """,
        agent=agent,
        expected_output="A comprehensive analysis of the technical drawing with all manufacturing specifications"
    )

def create_material_selection_task(agent, context=None):
    """Tạo task cho việc lựa chọn vật liệu tối ưu"""
    
    return Task(
        description="""
        Based on the design analysis, recommend optimal materials for the manufacturing process:
        1. Analyze the specified material requirements from the drawing
        2. Evaluate if the specified material is optimal for the application
        3. Recommend alternative materials that could improve:
           - Performance characteristics (strength, durability, etc.)
           - Manufacturability (machinability, formability, etc.)
           - Cost-efficiency
           - Sustainability and environmental impact
        4. Provide detailed justification for each recommendation
        5. Consider material availability and supply chain factors
        
        IMPORTANT FILE ACCESS RESTRICTIONS:
        
        1. DO NOT attempt to access ANY files EXCEPT for the specific part catalog at:
           "materials_catalogs/SC-2023-A001.txt"
           
        2. DO NOT attempt to access these files (they DO NOT exist):
           - "materials_catalogs/materials_summary.txt"
           - "materials_catalogs/AISI_4140.txt"
           - "materials_catalogs/alloy_steel.txt"
           - "/path/to/document.txt"
           - Any other file not explicitly listed in point #1
        
        3. If the specific catalog file "materials_catalogs/SC-2023-A001.txt" is not found, 
           DO NOT try other files - instead, rely solely on your internal knowledge and 
           the information provided in the context from the Design Analyzer's output.
        
        4. Use ONLY the information provided directly in your context from the Design Analyzer
           and the single authorized catalog file.
        
        5. If you need information that is not available in the context or the authorized catalog file,
           make reasonable assumptions based on standard material properties rather than trying to
           access other unauthorized files.
        
        Your recommendations should be practical and consider both technical and economic factors.
        """,
        agent=agent,
        expected_output="A detailed material selection analysis with optimal recommendations and alternatives",
        context=context
    )

def create_tooling_selection_task(agent, context=None):
    """Tạo task cho việc lựa chọn dụng cụ gia công"""
    
    return Task(
        description="""
        Based on the design analysis and material selection, recommend optimal tooling and fixtures for the manufacturing process:
        1. Analyze the part geometry, features, dimensions, and tolerances from the design
        2. Consider the selected material properties when recommending cutting tools
        3. Recommend specific tooling for each manufacturing operation:
           - Cutting tools (inserts, end mills, drills, reamers, etc.)
           - Tool holders and adapters
           - Fixtures and workholding devices
           - Measuring instruments for quality control
        4. Specify optimal cutting parameters for each tool (speeds, feeds, depths of cut)
        5. Provide justification for each tooling recommendation, considering:
           - Surface finish requirements
           - Dimensional accuracy and tolerance needs
           - Production efficiency and cycle time
           - Tool life and cost-effectiveness
        6. Suggest any specialized tooling needs for complex features
        
        IMPORTANT FILE ACCESS RESTRICTIONS:
        
        1. DO NOT attempt to access ANY files EXCEPT for the specific part catalog at:
           "tooling_catalogs/SC-2023-A001.txt"
           
        2. DO NOT attempt to access these files (they DO NOT exist):
           - "tooling_catalogs/tooling_summary.txt"
           - "tooling_catalogs/cutting_tools.txt"
           - "tooling_catalogs/fixtures.txt"
           - "/material_selection_analysis.txt"
           - Any other file not explicitly listed in point #1
        
        3. If the specific catalog file "tooling_catalogs/SC-2023-A001.txt" is not found, 
           DO NOT try other files - instead, rely solely on your internal knowledge and 
           the information provided in the context from previous agents.
        
        4. Use ONLY the information provided directly in your context from previous agents and 
           the single authorized catalog file.
        
        5. If you need information that is not available in the context or the authorized catalog file,
           make reasonable assumptions based on standard tooling knowledge rather than trying to
           access other unauthorized files.
        
        Your recommendations should be practical, considering both technical performance and economic factors.
        Be specific with tool designations, materials, coatings, and geometries where applicable.
        """,
        agent=agent,
        expected_output="A comprehensive tooling plan with specific recommendations for each manufacturing operation",
        context=context
    )

def create_process_plan_task(agent, context=None):
    """Tạo task cho việc lập quy trình gia công"""
    
    return Task(
        description="""
        Based on the design analysis, material selection, and tooling recommendations, develop a detailed manufacturing process plan:
        1. Determine the sequence of operations appropriate for the selected materials and tooling
        2. Specify machining methods for each feature, considering material properties and available tools
        3. Incorporate the recommended tooling and fixtures into each process step
        4. Apply the suggested cutting parameters (speeds, feeds, depths of cut) in the process
        5. Calculate estimated machining times based on material characteristics and cutting parameters
        6. Identify potential manufacturing challenges related to material properties or tooling limitations
        
        IMPORTANT FILE ACCESS RESTRICTIONS:
        
        1. DO NOT attempt to access ANY files EXCEPT for the specific process standards at:
           "process_standards/SC-2023-A001.txt"
           
        2. DO NOT attempt to access these files (they DO NOT exist):
           - "tooling_recommendations.txt" (with or without leading slash)
           - "/tooling_recommendations.txt" (with leading slash)
           - "detailed_tooling_recommendations.txt" (with or without leading slash)
           - "/detailed_tooling_recommendations.txt" (with leading slash)
           - "process_standards/general_machining.txt"
           - "process_standards/cutting_parameters.txt"
           - "/path/to/file.txt"
           - Any other file not explicitly listed in point #1
        
        3. If the specific standards file "process_standards/SC-2023-A001.txt" is not found, 
           DO NOT try other files - instead, rely solely on your internal knowledge and 
           the information provided in the context from previous agents.
        
        4. Use ONLY the information provided directly in your context from previous agents and 
           the single authorized standards file.
        
        5. If you need information that is not available in the context or the authorized standards file,
           make reasonable assumptions based on standard manufacturing practices rather than trying to
           access other unauthorized files.
        
        Integrate the design specifications, material recommendations, and tooling suggestions into a 
        cohesive process plan. Your plan should be detailed enough for shop floor implementation, with 
        clear instructions on:
        - What operations to perform in what sequence
        - Which tools to use for each operation
        - How to set up the machine and workholding for each step
        - What process parameters to apply
        - Expected outcomes for each operation
        """,
        agent=agent,
        expected_output="A step-by-step manufacturing process plan",
        context=context
    )

def create_quality_review_task(agent, context=None):
    """Tạo task cho việc kiểm tra chất lượng"""
    
    return Task(
        description="""
        Review the manufacturing process plan and perform a quality analysis:
        1. Identify critical quality checkpoints, considering material properties and tooling selections
        2. Suggest inspection methods and equipment appropriate for the selected materials and manufacturing processes
        3. Highlight potential quality risks related to material characteristics, tooling choices, or process parameters
        4. Recommend preventive actions, taking material properties and tooling capabilities into account
        5. Evaluate if the suggested tooling and fixtures will produce parts that meet the required specifications
        6. Verify that the selected cutting parameters will achieve the required surface finishes and dimensional accuracy
        
        IMPORTANT FILE ACCESS RESTRICTIONS:
        
        1. DO NOT attempt to access ANY files EXCEPT for the specific quality standards at:
           "quality_standards/SC-2023-A001.txt"
           
        2. DO NOT attempt to access these files (they DO NOT exist):
           - "quality_standards/general_standards.txt"
           - "quality_standards/inspection_methods.txt"
           - "quality_standards/acceptance_criteria.txt"
           - "/path/to/manufacturing_plan_document.txt"
           - Any other file not explicitly listed in point #1
        
        3. If the specific standards file "quality_standards/SC-2023-A001.txt" is not found, 
           DO NOT try other files - instead, rely solely on your internal knowledge and 
           the information provided in the context from previous agents.
        
        4. Use ONLY the information provided directly in your context from previous agents and 
           the single authorized standards file.
        
        5. If you need information that is not available in the context or the authorized standards file,
           make reasonable assumptions based on standard quality practices rather than trying to
           access other unauthorized files.
        
        Your review should be thorough and focus on maintaining quality throughout the process,
        with special attention to how material properties, tooling choices, and process parameters 
        collectively affect quality outcomes.
        """,
        agent=agent,
        expected_output="A quality assessment report with recommendations",
        context=context
    )

def create_finalize_plan_task(agent, context=None):
    """Tạo task cho việc hoàn thiện quy trình"""
    
    return Task(
        description="""
        Integrate all inputs to create a final manufacturing plan:
        1. Synthesize the process plan, material selection recommendations, tooling selections, and quality assessments
        2. Finalize material selections based on performance, cost, and manufacturability trade-offs
        3. Confirm tooling choices and cutting parameters for each operation
        4. Optimize operations sequence and parameters for the selected materials and tooling
        5. Create a comprehensive bill of materials including raw materials and tooling requirements
        6. Develop a detailed timeline with milestones that accounts for material procurement, tooling setup, and processing
        7. Assign responsibilities for each process step
        8. Create a final document that can be used to guide production
        
        IMPORTANT: Do NOT attempt to access any external files or paths. All the information you need is 
        already provided in the context from previous agents (Design Analyzer, Materials Engineer, Tooling Specialist, 
        Process Planner, and Quality Assurance Engineer). 
        
        Use ONLY the information directly provided in the context from the previous tasks. Do not try to access
        external files like "/path/to/manufacturing_plan_document.txt", "/material_selection_analysis.txt", etc.
        
        Address any conflicting recommendations between material selection, tooling choices, and manufacturing processes,
        and make clear decisions, justifying trade-offs when necessary.
        
        Your final plan should be comprehensive and ready for implementation, with all elements 
        (materials, tooling, processes, and quality measures) integrated into a cohesive strategy.
        """,
        agent=agent,
        expected_output="A finalized manufacturing plan ready for implementation",
        context=context
    ) 