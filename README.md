# Manufacturing Process Planning System

A multi-agent AI system for automatically generating detailed manufacturing process plans from technical drawings.

## Overview

This system uses a team of specialized AI agents to analyze technical drawings and create comprehensive manufacturing process plans. Each agent has specific expertise and responsibilities in the manufacturing planning workflow.

## System Architecture

The system uses the following specialized agents:

1. **Design Analysis Engineer** - Analyzes technical drawings to extract manufacturing specifications
2. **Materials Engineer** - Selects optimal materials based on design requirements and performance criteria
3. **Tooling Specialist** - Recommends appropriate tooling and fixtures for manufacturing operations
4. **Manufacturing Process Planner** - Creates detailed process plans and operation sequences
5. **Quality Assurance Engineer** - Evaluates manufacturing plans for quality and standards compliance
6. **Manufacturing Project Manager** - Coordinates between agents and synthesizes the final manufacturing plan

## Directory Structure

```
├── agents.py                  # Agent definitions and task configurations
├── config.py                  # Configuration settings
├── main.py                    # Main program entry point
├── utils.py                   # Utility functions
├── drawings/                  # Technical drawings input files
├── data/                      # Output directory for generated manufacturing plans
├── materials_catalogs/        # Reference materials for the Materials Engineer agent
│   └── SC-2023-A001.txt       # Materials catalog for shaft coupling SC-2023-A001
├── tooling_catalogs/          # Reference materials for the Tooling Specialist agent
│   └── SC-2023-A001.txt       # Tooling catalog for shaft coupling SC-2023-A001
├── process_standards/         # Reference materials for the Manufacturing Process Planner agent
│   └── SC-2023-A001.txt       # Process standards for shaft coupling SC-2023-A001
└── quality_standards/         # Reference materials for the Quality Assurance Engineer agent
    └── SC-2023-A001.txt       # Quality standards for shaft coupling SC-2023-A001
```

## Usage

Run the system with a drawing file using:

```bash
python main.py --drawing drawings/your_drawing_file.txt --verbose
```

Example:

```bash
python main.py --drawing drawings/Shaft_Coupling_Design_Requirements_Final_v21.txt --verbose
```

The system will:

1. Analyze the technical drawing
2. Select optimal materials
3. Recommend appropriate tooling
4. Create a detailed manufacturing process plan
5. Evaluate the plan for quality compliance
6. Generate a final comprehensive manufacturing plan

Output will be saved to the `data/` directory.

## Reference Catalogs

The system uses specialized reference catalogs for different agents:

- **Materials Catalogs**: Contain material specifications and alternatives
- **Tooling Catalogs**: Provide tooling recommendations and parameters
- **Process Standards**: Define operation sequences and machining parameters
- **Quality Standards**: Specify inspection methods and acceptance criteria

These catalogs are named according to the part number (e.g., SC-2023-A001.txt).

## Requirements

- Python 3.8+
- OpenAI API key (set in config.py)
- LangChain
- CrewAI

## License

[Add your license information here] 