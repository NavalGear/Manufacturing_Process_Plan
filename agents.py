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

# Define agent types
AGENT_TYPES = {
    "DESIGN_ANALYZER": "design_analyzer",
    "MATERIALS_ENGINEER": "materials_engineer",
    "TOOLING_SPECIALIST": "tooling_specialist",
    "PROCESS_PLANNER": "process_planner",
    "QUALITY_ENGINEER": "quality_engineer",
    "PROJECT_MANAGER": "project_manager"
}

# Configure the catalog mappings
CATALOG_CONFIG = {
    AGENT_TYPES["MATERIALS_ENGINEER"]: {
        "catalog_path": "materials_catalogs/SC-2023-A001.txt",
        "forbidden_files": [
            "materials_catalogs/materials_summary.txt",
            "materials_catalogs/AISI_4140.txt",
            "materials_catalogs/alloy_steel.txt"
        ],
        "fallback_guidance": "standard material properties"
    },
    AGENT_TYPES["TOOLING_SPECIALIST"]: {
        "catalog_path": "tooling_catalogs/SC-2023-A001.txt",
        "forbidden_files": [
            "tooling_catalogs/tooling_summary.txt",
            "tooling_catalogs/cutting_tools.txt",
            "tooling_catalogs/fixtures.txt"
        ],
        "fallback_guidance": "standard tooling knowledge"
    },
    AGENT_TYPES["PROCESS_PLANNER"]: {
        "catalog_path": "process_standards/SC-2023-A001.txt",
        "forbidden_files": [
            "tooling_recommendations.txt (with or without leading slash)",
            "/tooling_recommendations.txt (with leading slash)",
            "detailed_tooling_recommendations.txt (with or without leading slash)",
            "/detailed_tooling_recommendations.txt (with leading slash)",
            "process_standards/general_machining.txt",
            "process_standards/cutting_parameters.txt"
        ],
        "fallback_guidance": "standard manufacturing practices"
    },
    AGENT_TYPES["QUALITY_ENGINEER"]: {
        "catalog_path": "quality_standards/SC-2023-A001.txt",
        "forbidden_files": [
            "quality_standards/general_standards.txt",
            "quality_standards/inspection_methods.txt",
            "quality_standards/acceptance_criteria.txt"
        ],
        "fallback_guidance": "standard quality practices"
    },
    AGENT_TYPES["PROJECT_MANAGER"]: {
        "catalog_path": "",  # Project manager doesn't have a specific catalog
        "forbidden_files": [
            "/path/to/manufacturing_plan_document.txt",
            "/material_selection_analysis.txt",
            "/tooling_recommendations.txt",
            "/process_plan.txt",
            "/quality_assessment.txt"
        ],
        "fallback_guidance": "standard project management practices"
    }
}

def generate_file_access_restrictions(agent_type: str) -> str:
    """
    Creates standardized file access restriction text for agent backstories and tasks
    
    Args:
        agent_type: The type of agent (materials_engineer, tooling_specialist, etc.)
        
    Returns:
        Formatted restriction text
    """
    if agent_type not in CATALOG_CONFIG:
        return ""
        
    config = CATALOG_CONFIG[agent_type]
    
    # Using raw string for the template to avoid escape sequence issues
    restriction_text = f"""
        IMPORTANT FILE ACCESS RESTRICTIONS:
        
        1. DO NOT attempt to access ANY files EXCEPT for the specific part catalog at:
           "{config['catalog_path']}"
           
        2. DO NOT attempt to access these files (they DO NOT exist):"""
    
    # Add forbidden files one by one
    for file in config['forbidden_files']:
        restriction_text += f"""
           - {file}"""
    
    # Continue with the rest of the template
    restriction_text += f"""
           - "/path/to/document.txt"
           - Any other file not explicitly listed in point #1
        
        3. If the specific catalog file "{config['catalog_path']}" is not found, 
           DO NOT try other files - instead, rely solely on your internal knowledge and 
           the information provided in the context from previous agents.
        
        4. Use ONLY the information provided directly in your context from previous agents and 
           the single authorized catalog file.
        
        5. If you need information that is not available in the context or the authorized catalog file,
           make reasonable assumptions based on {config['fallback_guidance']} rather than trying to
           access other unauthorized files."""
           
    return restriction_text

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

def create_agent(agent_type, role, goal, backstory_content, allow_delegation=False, temperature=None):
    """
    Utility function to create an agent with standard configurations and file restrictions
    
    Args:
        agent_type: Type of agent from AGENT_TYPES
        role: Agent's role description
        goal: Agent's goal
        backstory_content: The agent-specific backstory content (before file restrictions)
        allow_delegation: Whether the agent can delegate tasks
        temperature: Optional temperature override for the agent's LLM
        
    Returns:
        Agent: Configured agent
    """
    # Generate file access restrictions if applicable
    file_restrictions = generate_file_access_restrictions(agent_type)
    
    # Combine backstory with restrictions
    full_backstory = f"{backstory_content}\n\n{file_restrictions}" if file_restrictions else backstory_content
    
    return Agent(
        role=role,
        goal=goal,
        backstory=full_backstory,
        verbose=config.VERBOSE,
        allow_delegation=allow_delegation,
        tools=[file_tool] if agent_type != AGENT_TYPES["PROJECT_MANAGER"] else [],
        llm=get_llm(temperature=temperature)
    )

# 1. Tác tử Phân Tích Bản Vẽ (Design Analyzer Agent)
def create_design_analyzer_agent():
    """Tạo tác tử phân tích bản vẽ kỹ thuật"""
    
    backstory_content = """You are an expert mechanical engineer specialized in CAD analysis and design interpretation.
    You can look at technical drawings and extract precise measurements, tolerances, material requirements, 
    and other manufacturing specifications. Your analysis forms the foundation for the entire 
    manufacturing process, so accuracy and completeness are critical.
    
    IMPORTANT: Do NOT attempt to access any external files using file paths. The drawing content 
    will be directly provided to you in your task description. Only use the information directly 
    given to you in the task description between the triple backticks."""
    
    return create_agent(
        agent_type=AGENT_TYPES["DESIGN_ANALYZER"],
        role="Design Analysis Engineer",
        goal="Analyze technical drawings to extract all relevant manufacturing specifications",
        backstory_content=backstory_content
    )

# 1.5 Tác tử Lựa Chọn Vật Liệu Tối Ưu (Material Selection Specialist)
def create_material_selection_agent():
    """Tạo tác tử lựa chọn vật liệu tối ưu"""
    
    backstory_content = """You are a highly skilled materials engineer with extensive knowledge of engineering 
    materials, their properties, and processing requirements. You understand the complex relationships 
    between material properties, manufacturing processes, costs, and sustainability factors.
    You can analyze design specifications to recommend the most suitable materials, suggest viable 
    alternatives, and optimize material selection to balance performance, manufacturability, cost, 
    and environmental considerations. Your expertise ensures that material choices will meet all 
    technical requirements while optimizing production efficiency and product quality."""
    
    return create_agent(
        agent_type=AGENT_TYPES["MATERIALS_ENGINEER"],
        role="Materials Engineer",
        goal="Select optimal materials for manufacturing based on design requirements, performance, cost, and manufacturability",
        backstory_content=backstory_content
    )

# 1.8 Tác tử Lựa Chọn Dụng Cụ Gia Công (Tooling Selection Specialist)
def create_tooling_selection_agent():
    """Tạo tác tử lựa chọn dụng cụ gia công"""
    
    backstory_content = """You are an expert tooling engineer with decades of experience in selecting and 
    optimizing cutting tools, fixtures, and machining parameters. You have deep knowledge of tool 
    geometries, materials, coatings, and their applications across various manufacturing processes 
    including turning, milling, drilling, grinding, and EDM. You understand how tool selection affects 
    surface finish, dimensional accuracy, cycle time, and production costs. Your expertise enables you 
    to recommend the most appropriate tooling solutions for specific manufacturing challenges, 
    considering factors such as workpiece material properties, geometric complexity, tolerance requirements, 
    and production volume. You stay current with the latest tooling technologies and can suggest innovative 
    solutions to improve manufacturing efficiency and quality."""
    
    return create_agent(
        agent_type=AGENT_TYPES["TOOLING_SPECIALIST"],
        role="Tooling Specialist",
        goal="Select optimal tooling and fixtures for manufacturing operations based on part geometry, material, and precision requirements",
        backstory_content=backstory_content
    )

# 2. Tác tử Lập Quy Trình Gia Công (Process Planning Agent)
def create_process_planning_agent():
    """Tạo tác tử lập quy trình gia công"""
    
    backstory_content = """You are a manufacturing process expert with decades of experience.
    You understand machining processes, tooling requirements, and can optimize production 
    for efficiency, cost-effectiveness, and quality. Your expertise covers turning, milling,
    drilling, grinding, and other common manufacturing processes. You can determine the optimal
    sequence of operations to create a part according to specifications."""
    
    return create_agent(
        agent_type=AGENT_TYPES["PROCESS_PLANNER"],
        role="Manufacturing Process Planner",
        goal="Create detailed manufacturing process plans based on design specifications",
        backstory_content=backstory_content
    )

# 3. Tác tử Kiểm Tra Chất Lượng (Quality Control Agent)
def create_quality_control_agent():
    """Tạo tác tử kiểm tra chất lượng"""
    
    backstory_content = """You are a meticulous quality assurance engineer with expertise in 
    mechanical manufacturing. You can detect potential issues in manufacturing plans,
    suggest quality control checkpoints, and ensure the final product will meet all specifications.
    You understand measurement techniques, inspection equipment, and quality standards
    such as ISO 9001."""
    
    return create_agent(
        agent_type=AGENT_TYPES["QUALITY_ENGINEER"],
        role="Quality Assurance Engineer",
        goal="Evaluate manufacturing plans to ensure they meet quality standards and specifications",
        backstory_content=backstory_content
    )

# 4. Tác tử Điều Phối (Orchestrator Agent)
def create_orchestrator_agent():
    """Tạo tác tử điều phối tổng thể"""
    
    backstory_content = """You are a seasoned manufacturing project manager who excels at
    coordinating cross-functional teams. You understand both the technical and business
    aspects of manufacturing, and can synthesize inputs from different specialists to
    create cohesive manufacturing plans. You seek to optimize processes for efficiency,
    quality, and cost-effectiveness."""
    
    return create_agent(
        agent_type=AGENT_TYPES["PROJECT_MANAGER"],
        role="Manufacturing Project Manager",
        goal="Coordinate the overall manufacturing process planning and optimize for quality and efficiency",
        backstory_content=backstory_content,
        allow_delegation=True,
        temperature=0.4  # Nhiệt độ cao hơn cho sự sáng tạo
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
    
    description_content = """
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
    
    Your recommendations should be practical and consider both technical and economic factors.
    """
    
    return create_task(
        agent_type=AGENT_TYPES["MATERIALS_ENGINEER"],
        agent=agent,
        description_content=description_content,
        expected_output="A detailed material selection analysis with optimal recommendations and alternatives",
        context=context
    )

def create_tooling_selection_task(agent, context=None):
    """Tạo task cho việc lựa chọn dụng cụ gia công"""
    
    description_content = """
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
    
    Your recommendations should be practical, considering both technical performance and economic factors.
    Be specific with tool designations, materials, coatings, and geometries where applicable.
    """
    
    return create_task(
        agent_type=AGENT_TYPES["TOOLING_SPECIALIST"],
        agent=agent,
        description_content=description_content,
        expected_output="A comprehensive tooling plan with specific recommendations for each manufacturing operation",
        context=context
    )

def create_process_plan_task(agent, context=None):
    """Tạo task cho việc lập quy trình gia công"""
    
    description_content = """
    Based on the design analysis, material selection, and tooling recommendations, develop a detailed manufacturing process plan:
    1. Determine the sequence of operations appropriate for the selected materials and tooling
    2. Specify machining methods for each feature, considering material properties and available tools
    3. Incorporate the recommended tooling and fixtures into each process step
    4. Apply the suggested cutting parameters (speeds, feeds, depths of cut) in the process
    5. Calculate estimated machining times based on material characteristics and cutting parameters
    6. Identify potential manufacturing challenges related to material properties or tooling limitations
    
    Integrate the design specifications, material recommendations, and tooling suggestions into a 
    cohesive process plan. Your plan should be detailed enough for shop floor implementation, with 
    clear instructions on:
    - What operations to perform in what sequence
    - Which tools to use for each operation
    - How to set up the machine and workholding for each step
    - What process parameters to apply
    - Expected outcomes for each operation
    """
    
    return create_task(
        agent_type=AGENT_TYPES["PROCESS_PLANNER"],
        agent=agent,
        description_content=description_content,
        expected_output="A step-by-step manufacturing process plan",
        context=context
    )

def create_quality_review_task(agent, context=None):
    """Tạo task cho việc kiểm tra chất lượng"""
    
    description_content = """
    Review the manufacturing process plan and perform a quality analysis:
    1. Identify critical quality checkpoints, considering material properties and tooling selections
    2. Suggest inspection methods and equipment appropriate for the selected materials and manufacturing processes
    3. Highlight potential quality risks related to material characteristics, tooling choices, or process parameters
    4. Recommend preventive actions, taking material properties and tooling capabilities into account
    5. Evaluate if the suggested tooling and fixtures will produce parts that meet the required specifications
    6. Verify that the selected cutting parameters will achieve the required surface finishes and dimensional accuracy
    
    Your review should be thorough and focus on maintaining quality throughout the process,
    with special attention to how material properties, tooling choices, and process parameters 
    collectively affect quality outcomes.
    """
    
    return create_task(
        agent_type=AGENT_TYPES["QUALITY_ENGINEER"],
        agent=agent,
        description_content=description_content,
        expected_output="A quality assessment report with recommendations",
        context=context
    )

def create_finalize_plan_task(agent, context=None):
    """Tạo task cho việc hoàn thiện quy trình"""
    
    description_content = """
    Integrate all inputs to create a final manufacturing plan:
    1. Synthesize the process plan, material selection recommendations, tooling selections, and quality assessments
    2. Finalize material selections based on performance, cost, and manufacturability trade-offs
    3. Confirm tooling choices and cutting parameters for each operation
    4. Optimize operations sequence and parameters for the selected materials and tooling
    5. Create a comprehensive bill of materials including raw materials and tooling requirements
    6. Develop a detailed timeline with milestones that accounts for material procurement, tooling setup, and processing
    7. Assign responsibilities for each process step
    8. Create a final document that can be used to guide production
    
    Address any conflicting recommendations between material selection, tooling choices, and manufacturing processes,
    and make clear decisions, justifying trade-offs when necessary.
    
    Your final plan should be comprehensive and ready for implementation, with all elements 
    (materials, tooling, processes, and quality measures) integrated into a cohesive strategy.
    """
    
    return create_task(
        agent_type=AGENT_TYPES["PROJECT_MANAGER"],
        agent=agent,
        description_content=description_content,
        expected_output="A finalized manufacturing plan ready for implementation",
        context=context
    )

def create_task(agent_type, agent, description_content, expected_output, context=None):
    """
    Utility function to create a task with standard configurations and file restrictions
    
    Args:
        agent_type: Type of agent from AGENT_TYPES
        agent: The agent that will perform this task
        description_content: The task-specific description content (before file restrictions)
        expected_output: Expected output description
        context: Optional context from previous tasks
        
    Returns:
        Task: Configured task
    """
    # Generate file access restrictions if applicable
    file_restrictions = generate_file_access_restrictions(agent_type)
    
    # Combine description with restrictions
    full_description = f"{description_content}\n\n{file_restrictions}" if file_restrictions else description_content
    
    return Task(
        description=full_description,
        agent=agent,
        expected_output=expected_output,
        context=context
    ) 