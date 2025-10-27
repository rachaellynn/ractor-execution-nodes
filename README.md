# Ractor Execution Nodes

This repository contains execution logic for Ractor agents. It provides specialized execution capabilities for Python, JavaScript, file operations, and web operations within the Ractor agent platform.

## Overview

This repository provides execution nodes that run within Ractor agents to handle:
- Python code execution with dependency management
- JavaScript/Node.js code execution
- File operations (read/write/create)
- Web operations (downloads, API calls)

## Architecture

### Agent Setup Code Generation
The system generates setup code for different node types:
- **Python nodes**: Install common packages (pandas, numpy, matplotlib, requests, boto3)
- **JavaScript nodes**: Initialize npm, install Express, axios, csv-parser
- **File operations**: Set up file handling environment
- **Web operations**: Configure HTTP clients and S3 access

### Execution Tracking
Each execution is tracked with detailed metrics:
- Individual tool call breakdown
- Execution time per operation
- Success/failure tracking with retry logic
- Data payload sizes (request/response)
- Files created and dependencies installed

## Usage with Ractor API

This code is designed to be deployed as agents within the Ractor platform (`https://githex-hyp-1.raworc.com`).

### Agent Creation
```python
agent_payload = {
    "name": "task-{task_id}-{node_type}",
    "setup": setup_code,  # Generated based on node_type
    "instructions": f"You are a {node_type} execution agent.",
    "prompt": "Execute the provided task and return detailed results."
}
```

### Task Execution
```python
input_payload = {
    "input": {
        "content": [{"type": "text", "content": task_description}]
    },
    "background": False
}
```

## Metrics and Benchmarking

The system tracks four key metrics:
1. **Tool calls**: Individual operations counted (pip installs, file writes, subprocess calls)
2. **Latency**: Network latency + computation time breakdown
3. **Success rate**: Success/failure tracking with retry attempts
4. **Data in transit**: Request/response payload sizes

### Tool Call Breakdown
- File operations (file reads/writes)
- Subprocess calls (python, node, etc.)
- Package installations (pip, npm)
- Network operations (downloads, API calls)

## File Structure

```
ractor-execution-nodes/
├── executor.py          # Core execution logic
├── metrics.py           # Payload tracking and metrics
├── requirements.txt     # Python dependencies
├── package.json         # Node.js dependencies
└── README.md           # This file
```

## Dependencies

### Python
- `asyncio` - Async execution
- `aiofiles` - Async file operations
- `httpx` - HTTP client for downloads
- `pydantic` - Data validation

### Node.js
- `express` - Web server framework
- `axios` - HTTP client
- `csv-parser` - CSV parsing

## Integration

This repository is used by the LangGraph orchestrator through the Ractor API:

1. **Agent Creation**: Setup code from this repo is sent to Ractor agents
2. **Task Execution**: Task descriptions are sent to agents via `/api/v0/agents/{name}/responses`
3. **Result Polling**: Orchestrator polls for completion and collects metrics
4. **Session Management**: Agents maintain persistent workspaces between tasks

## Benchmarking

The execution nodes support the benchmarking scenarios:

### Example Use Cases
- Install and use data science packages (pandas, numpy, matplotlib)
- Create Node.js projects with Express servers
- Download and process CSV data files
- Build multi-step data processing pipelines

## Local Development

For local testing, the execution logic can be run standalone:

```python
from executor import RactorNodeExecutor

executor = RactorNodeExecutor()
result = await executor.execute_python({
    "task_id": "test-123",
    "task_description": "Install pandas and create a DataFrame",
    "node_type": "python"
})
```

## License

MIT License - see LICENSE file for details.