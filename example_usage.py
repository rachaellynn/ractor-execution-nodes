"""
Example usage of Ractor Execution Nodes.

This shows how the execution logic would be used within Ractor agents.
"""
import asyncio
from executor import (
    RactorExecutionAgent,
    generate_python_setup,
    generate_javascript_setup,
    generate_file_ops_setup,
    generate_web_ops_setup
)


async def example_python_execution():
    """Example of Python task execution"""
    print("=== Python Execution Example ===")

    agent = RactorExecutionAgent()

    # Example task: Install pandas and create DataFrame
    task_description = "Install pandas, numpy, matplotlib and create a simple DataFrame with coffee shop data"

    result = await agent.execute_task(task_description, node_type="python")

    print(f"Success: {result['success']}")
    print(f"Execution time: {result['execution_time_ms']:.2f}ms")
    print(f"Tool calls: {result['tool_calls_count']}")
    print(f"Files created: {result['files_created']}")
    print(f"Dependencies: {result['dependencies_installed']}")
    print(f"Result: {result.get('result', '')[:200]}...")  # First 200 chars


async def example_javascript_execution():
    """Example of JavaScript task execution"""
    print("\n=== JavaScript Execution Example ===")

    agent = RactorExecutionAgent()

    # Example task: Create Express server
    task_description = "Create a Node.js project with npm init, install express, and create a simple server with a /hello route"

    result = await agent.execute_task(task_description, node_type="javascript")

    print(f"Success: {result['success']}")
    print(f"Execution time: {result['execution_time_ms']:.2f}ms")
    print(f"Tool calls: {result['tool_calls_count']}")
    print(f"Files created: {result['files_created']}")
    print(f"Result: {result.get('result', '')[:200]}...")


async def example_web_operations():
    """Example of web operations"""
    print("\n=== Web Operations Example ===")

    agent = RactorExecutionAgent()

    # Example task: Download CSV data
    task_description = "Download GDP data from https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv"

    result = await agent.execute_task(task_description, node_type="web_ops")

    print(f"Success: {result['success']}")
    print(f"Execution time: {result['execution_time_ms']:.2f}ms")
    print(f"Tool calls: {result['tool_calls_count']}")
    print(f"Files created: {result['files_created']}")
    print(f"Result: {result.get('result', '')}")


def show_setup_code_examples():
    """Show the setup code that would be deployed to Ractor agents"""
    print("\n=== Setup Code Examples ===")

    print("Python Setup Code:")
    print("-" * 40)
    print(generate_python_setup()[:200] + "...")

    print("\nJavaScript Setup Code:")
    print("-" * 40)
    print(generate_javascript_setup()[:200] + "...")

    print("\nFile Operations Setup Code:")
    print("-" * 40)
    print(generate_file_ops_setup()[:200] + "...")

    print("\nWeb Operations Setup Code:")
    print("-" * 40)
    print(generate_web_ops_setup()[:200] + "...")


async def main():
    """Run all examples"""
    print("Ractor Execution Nodes - Example Usage")
    print("=" * 50)

    # Show setup codes
    show_setup_code_examples()

    # Run execution examples
    try:
        await example_python_execution()
        await example_javascript_execution()
        await example_web_operations()
    except Exception as e:
        print(f"Example execution failed: {e}")
        print("Note: Some examples may fail without proper environment setup")

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())