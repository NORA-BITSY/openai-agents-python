# OpenAI Agents SDK - Comprehensive User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Core Concepts](#core-concepts)
4. [Agent Configuration](#agent-configuration)
5. [Tools](#tools)
6. [Handoffs](#handoffs)
7. [Guardrails](#guardrails)
8. [Sessions & Memory](#sessions--memory)
9. [Running Agents](#running-agents)
10. [Tracing & Debugging](#tracing--debugging)
11. [Examples & Patterns](#examples--patterns)
12. [Advanced Features](#advanced-features)
13. [API Reference](#api-reference)
14. [Troubleshooting](#troubleshooting)

---

## Introduction

The OpenAI Agents SDK is a lightweight, production-ready framework for building multi-agent AI workflows. It enables you to create sophisticated agentic applications with minimal abstractions while providing powerful features like handoffs, guardrails, sessions, and built-in tracing.

### Key Features
- **Agent Loop**: Built-in agent loop handling tool calls and LLM interactions
- **Python-First**: Uses native Python features for orchestration
- **Handoffs**: Delegate tasks between specialized agents
- **Guardrails**: Input/output validation running in parallel
- **Sessions**: Automatic conversation history management
- **Function Tools**: Convert any Python function into an agent tool
- **Tracing**: Built-in visualization and debugging capabilities

---

## Installation & Setup

### Basic Installation

```bash
pip install openai-agents
```

### Environment Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

### Development Installation

For contributing to the SDK:

```bash
git clone https://github.com/NORA-BITSY/openai-agents-python
cd openai-agents-python
make format  # Format code
make lint    # Lint code
make mypy    # Type check
make tests   # Run tests
```

---

## Core Concepts

### 1. Agents
Agents are LLMs equipped with instructions and tools. They represent the core building blocks of your application.

### 2. Handoffs
Allow agents to delegate tasks to other specialized agents, enabling modular architectures.

### 3. Guardrails
Run validation checks in parallel with agents to ensure safe and appropriate responses.

### 4. Sessions
Automatically maintain conversation history across multiple agent runs.

### 5. Tools
Enable agents to take actions - from function calls to web searches to computer use.

---

## Agent Configuration

### Basic Agent Setup

```python
from agents import Agent, ModelSettings

agent = Agent(
    name="Customer Support Assistant",
    instructions="You are a helpful customer support agent. Be concise and professional.",
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.1,
        top_p=0.9
    )
)
```

### Required Properties

- **`name`**: String identifier for your agent
- **`instructions`**: System prompt or developer message

### Optional Properties

- **`model`**: LLM model to use (defaults to OpenAI models)
- **`model_settings`**: Temperature, top_p, and other model parameters
- **`tools`**: List of tools the agent can use
- **`handoffs`**: Other agents this agent can delegate to
- **`guardrails`**: Input/output validation functions
- **`output_type`**: Structured output type (Pydantic models, etc.)

### Context Types

Agents can be typed with context for dependency injection:

```python
from dataclasses import dataclass
from agents import Agent

@dataclass
class UserContext:
    user_id: str
    is_premium: bool
    
    async def get_user_data(self):
        # Fetch user-specific data
        return {}

agent = Agent[UserContext](
    name="Personalized Assistant",
    instructions="Provide personalized assistance based on user context"
)
```

### Dynamic Instructions

Instructions can be generated dynamically based on context:

```python
def get_instructions(ctx, agent):
    user = ctx.context
    if user.is_premium:
        return "You are a premium support agent with access to all features."
    return "You are a standard support agent."

agent = Agent[UserContext](
    name="Dynamic Agent",
    instructions=get_instructions
)
```

### Output Types

Specify structured output formats:

```python
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = Agent(
    name="Calendar Parser",
    instructions="Extract calendar events from text",
    output_type=CalendarEvent
)
```

### Agent Cloning

Create variations of existing agents:

```python
base_agent = Agent(
    name="Assistant",
    instructions="You are helpful",
    model="gpt-4o"
)

specialized_agent = base_agent.clone(
    name="Math Tutor",
    instructions="You are a math tutor specializing in algebra"
)
```

---

## Tools

Tools enable agents to take actions beyond text generation. The SDK supports three types of tools:

### 1. Hosted Tools

OpenAI-provided tools that run on their servers:

```python
from agents import Agent, WebSearchTool, FileSearchTool, ComputerTool

agent = Agent(
    name="Research Assistant",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=5,
            vector_store_ids=["vs_123"]
        ),
        ComputerTool()
    ]
)
```

#### Available Hosted Tools:
- **WebSearchTool**: Search the web
- **FileSearchTool**: Search vector stores
- **ComputerTool**: Automate computer tasks
- **CodeInterpreterTool**: Execute code in sandbox
- **ImageGenerationTool**: Generate images
- **LocalShellTool**: Run shell commands

### 2. Function Tools

Convert any Python function into a tool:

```python
from agents import function_tool
from typing import TypedDict

class Location(TypedDict):
    lat: float
    lng: float

@function_tool
async def get_weather(location: Location, units: str = "celsius") -> str:
    """Get weather for a location.
    
    Args:
        location: Geographic coordinates
        units: Temperature units (celsius/fahrenheit)
    """
    # Implementation here
    return f"Weather at {location}: 22°C, sunny"

@function_tool(name_override="read_data")
def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read file contents.
    
    Args:
        path: File path to read
        encoding: File encoding
    """
    with open(path, 'r', encoding=encoding) as f:
        return f.read()

agent = Agent(
    name="Data Assistant",
    tools=[get_weather, read_file]
)
```

#### Function Tool Features:
- **Automatic Schema Generation**: From type annotations
- **Docstring Parsing**: For descriptions and parameter docs
- **Async/Sync Support**: Both function types supported
- **Context Access**: Optional first parameter for run context
- **Custom Names**: Override function names for tools

### 3. Agents as Tools

Use one agent as a tool for another:

```python
translator_agent = Agent(
    name="Translator",
    instructions="Translate text to the specified language"
)

main_agent = Agent(
    name="Multilingual Assistant", 
    tools=[translator_agent]
)
```

### Tool Error Handling

Handle errors gracefully in function tools:

```python
@function_tool
def divide_numbers(a: float, b: float) -> str:
    """Divide two numbers."""
    try:
        result = a / b
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Cannot divide by zero"
    except Exception as e:
        return f"Error: {str(e)}"
```

---

## Handoffs

Handoffs enable agents to delegate tasks to specialized sub-agents.

### Basic Handoffs

```python
from agents import Agent

# Specialized agents
booking_agent = Agent(
    name="Booking Agent",
    instructions="Handle flight bookings and reservations"
)

refund_agent = Agent(
    name="Refund Agent", 
    instructions="Process refund requests"
)

# Main triage agent
triage_agent = Agent(
    name="Customer Service",
    instructions="""
    Help customers with their requests.
    - For bookings, handoff to Booking Agent
    - For refunds, handoff to Refund Agent
    """,
    handoffs=[booking_agent, refund_agent]
)
```

### Advanced Handoffs

Use the `handoff()` function for customization:

```python
from agents import Agent, handoff, RunContextWrapper
from pydantic import BaseModel

class EscalationData(BaseModel):
    reason: str
    priority: str

async def on_escalation(ctx: RunContextWrapper, data: EscalationData):
    print(f"Escalation: {data.reason} (Priority: {data.priority})")
    # Log escalation, send notifications, etc.

escalation_agent = Agent(name="Escalation Agent")

advanced_handoff = handoff(
    agent=escalation_agent,
    tool_name_override="escalate_issue",
    tool_description_override="Escalate complex issues to senior support",
    on_handoff=on_escalation,
    input_type=EscalationData
)

agent = Agent(
    name="Support Agent",
    handoffs=[advanced_handoff]
)
```

### Input Filters

Control what information is passed to handoff agents:

```python
from agents import handoff
from agents.extensions import handoff_filters

# Remove all tool calls from history
clean_handoff = handoff(
    agent=specialized_agent,
    input_filter=handoff_filters.remove_all_tools
)

# Custom filter
def custom_filter(input_data):
    # Modify the input data before passing to next agent
    return input_data

filtered_handoff = handoff(
    agent=specialized_agent,
    input_filter=custom_filter
)
```

---

## Guardrails

Guardrails run validation checks in parallel with agents to ensure safe, appropriate responses.

### Input Guardrails

Validate user input before processing:

```python
from agents import Agent, input_guardrail, GuardrailFunctionOutput
from pydantic import BaseModel

class ContentModerationResult(BaseModel):
    is_appropriate: bool
    reason: str

moderation_agent = Agent(
    name="Content Moderator",
    instructions="Check if content is appropriate for our service",
    output_type=ContentModerationResult
)

@input_guardrail
async def content_guardrail(ctx, agent, input_data) -> GuardrailFunctionOutput:
    result = await Runner.run(moderation_agent, input_data)
    
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_appropriate
    )

protected_agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    input_guardrails=[content_guardrail]
)
```

### Output Guardrails

Validate agent outputs before returning to user:

```python
@output_guardrail  
async def response_guardrail(ctx, agent, output) -> GuardrailFunctionOutput:
    # Check if output contains sensitive information
    has_sensitive_data = check_for_pii(output)
    
    return GuardrailFunctionOutput(
        output_info={"contains_pii": has_sensitive_data},
        tripwire_triggered=has_sensitive_data
    )

secure_agent = Agent(
    name="Secure Assistant",
    output_guardrails=[response_guardrail]
)
```

### Handling Guardrail Exceptions

```python
from agents import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

try:
    result = await Runner.run(protected_agent, user_input)
    print(result.final_output)
except InputGuardrailTripwireTriggered as e:
    print("Input blocked by guardrail:", e.guardrail_result.output_info)
except OutputGuardrailTripwireTriggered as e:
    print("Output blocked by guardrail:", e.guardrail_result.output_info)
```

---

## Sessions & Memory

Sessions automatically maintain conversation history across multiple agent runs.

### Basic Session Usage

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(
    name="Chat Assistant",
    instructions="You are a helpful assistant"
)

# Create session with unique ID
session = SQLiteSession("user_123")

# First interaction
result = await Runner.run(
    agent,
    "What's the capital of France?",
    session=session
)
print(result.final_output)  # "Paris"

# Second interaction - remembers context
result = await Runner.run(
    agent, 
    "What's its population?",
    session=session
)
print(result.final_output)  # "Approximately 2.2 million"
```

### Session Operations

```python
# Get conversation history
items = await session.get_items()

# Add items manually
await session.add_items([
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
])

# Remove last item (useful for corrections)
last_item = await session.pop_item()

# Clear entire session
await session.clear_session()
```

### Custom Session Implementation

Implement your own session storage:

```python
from agents.memory import Session
from typing import List, Dict, Any

class RedisSession(Session):
    def __init__(self, session_id: str, redis_client):
        self.session_id = session_id
        self.redis = redis_client
    
    async def get_items(self) -> List[Dict[str, Any]]:
        # Retrieve from Redis
        pass
    
    async def add_items(self, items: List[Dict[str, Any]]) -> None:
        # Store to Redis
        pass
    
    async def pop_item(self) -> Dict[str, Any] | None:
        # Remove last item from Redis
        pass
        
    async def clear_session(self) -> None:
        # Clear Redis key
        pass
```

---

## Running Agents

### Basic Execution

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are helpful"
)

# Async execution
result = await Runner.run(agent, "Hello!")
print(result.final_output)

# Sync execution
result = Runner.run_sync(agent, "Hello!")
print(result.final_output)
```

### Streaming Execution

```python
# Stream agent responses
async for event in Runner.run_streamed(agent, "Tell me a story"):
    if event.type == "text_chunk":
        print(event.data, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[Tool: {event.data.name}]")

# Get final result
result = await stream_result.get_result()
```

### Run Configuration

```python
from agents import RunConfig

config = RunConfig(
    max_turns=10,
    model="gpt-4o",
    tracing_disabled=False,
    trace_include_sensitive_data=True
)

result = await Runner.run(
    agent,
    "Complex task",
    run_config=config
)
```

### The Agent Loop

The runner automatically handles:

1. **LLM Generation**: Call the model with current input
2. **Tool Execution**: Run any tool calls and append results  
3. **Handoff Processing**: Switch agents when handoffs occur
4. **Turn Management**: Prevent infinite loops with max_turns
5. **Result Formatting**: Return structured results

### Exception Handling

```python
from agents import MaxTurnsExceeded, AgentError

try:
    result = await Runner.run(agent, "Complex task")
except MaxTurnsExceeded:
    print("Agent exceeded maximum turns")
except AgentError as e:
    print(f"Agent error: {e}")
```

---

## Tracing & Debugging

The SDK includes comprehensive tracing for debugging and monitoring.

### Automatic Tracing

Tracing is enabled by default and captures:
- Agent executions
- LLM generations  
- Tool calls
- Handoffs
- Guardrails
- Audio processing

### Custom Traces

```python
from agents import trace, custom_span

async def complex_workflow():
    with trace("Multi-Agent Workflow") as t:
        # First agent
        result1 = await Runner.run(agent1, "Task 1")
        
        with custom_span("Data Processing") as span:
            processed_data = process_data(result1.final_output)
            span.add_metadata({"records_processed": len(processed_data)})
        
        # Second agent
        result2 = await Runner.run(agent2, processed_data)
        
        return result2
```

### Trace Configuration

```python
# Disable tracing globally
import os
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

# Disable for specific run
config = RunConfig(tracing_disabled=True)
result = await Runner.run(agent, "Input", run_config=config)

# Hide sensitive data
config = RunConfig(trace_include_sensitive_data=False)
```

### Custom Trace Processors

```python
from agents.tracing import add_trace_processor

class CustomTraceProcessor:
    async def process_trace(self, trace_data):
        # Send to your monitoring system
        print(f"Trace: {trace_data.trace_id}")
        
    async def process_span(self, span_data):
        # Process individual spans
        print(f"Span: {span_data.span_type}")

add_trace_processor(CustomTraceProcessor())
```

---

## Examples & Patterns

### 1. Deterministic Workflows

```python
from agents import Agent, Runner

async def structured_workflow(user_input: str):
    # Step 1: Analysis
    analyzer = Agent(
        name="Analyzer",
        instructions="Analyze the input and extract key requirements"
    )
    analysis = await Runner.run(analyzer, user_input)
    
    # Step 2: Planning
    planner = Agent(
        name="Planner", 
        instructions="Create a detailed plan based on the analysis"
    )
    plan = await Runner.run(planner, analysis.final_output)
    
    # Step 3: Execution
    executor = Agent(
        name="Executor",
        instructions="Execute the plan step by step"
    )
    result = await Runner.run(executor, plan.final_output)
    
    return result
```

### 2. Parallel Agent Execution

```python
import asyncio

async def parallel_processing(tasks: list[str]):
    # Create specialized agents
    agents = [
        Agent(name=f"Worker {i}", instructions="Process the given task")
        for i in range(len(tasks))
    ]
    
    # Run in parallel
    results = await asyncio.gather(*[
        Runner.run(agent, task)
        for agent, task in zip(agents, tasks)
    ])
    
    return [r.final_output for r in results]
```

### 3. LLM-as-a-Judge Pattern

```python
async def judge_responses(question: str, responses: list[str]):
    judge = Agent(
        name="Judge",
        instructions="""
        Evaluate responses for accuracy, helpfulness, and clarity.
        Rate each response 1-10 and provide reasoning.
        """,
        output_type=list[ResponseRating]
    )
    
    evaluation_input = f"""
    Question: {question}
    
    Responses to evaluate:
    {chr(10).join(f"{i+1}. {r}" for i, r in enumerate(responses))}
    """
    
    result = await Runner.run(judge, evaluation_input)
    return result.final_output
```

### 4. Customer Service System

```python
from agents import Agent, function_tool

@function_tool
async def lookup_order(order_id: str) -> dict:
    """Look up order details by ID."""
    # Database lookup
    return {"order_id": order_id, "status": "shipped"}

@function_tool  
async def process_refund(order_id: str, reason: str) -> str:
    """Process a refund request."""
    # Refund logic
    return f"Refund processed for order {order_id}"

# Specialized agents
order_agent = Agent(
    name="Order Agent",
    instructions="Help customers with order inquiries",
    tools=[lookup_order]
)

refund_agent = Agent(
    name="Refund Agent", 
    instructions="Process refund requests",
    tools=[process_refund]
)

# Main triage agent
triage_agent = Agent(
    name="Customer Service",
    instructions="""
    You are a customer service agent. Route requests appropriately:
    - Order questions → Order Agent
    - Refund requests → Refund Agent
    - General questions → handle directly
    """,
    handoffs=[order_agent, refund_agent]
)
```

---

## Advanced Features

### Model Providers

Use non-OpenAI models:

```python
from agents import Agent, set_default_openai_client
from openai import AsyncOpenAI

# Use alternative provider with OpenAI-compatible API
client = AsyncOpenAI(
    base_url="https://api.anthropic.com/v1",
    api_key="your-key"
)
set_default_openai_client(client)

agent = Agent(
    name="Claude Agent",
    model="claude-3-sonnet",
    instructions="You are Claude"
)
```

### Lifecycle Hooks

Monitor agent lifecycle events:

```python
from agents import AgentHooks

class CustomHooks(AgentHooks):
    async def on_start(self, ctx, agent):
        print(f"Agent {agent.name} starting")
        
    async def on_end(self, ctx, agent, output):
        print(f"Agent {agent.name} finished")
        
    async def on_tool_start(self, ctx, agent, tool):
        print(f"Calling tool: {tool.name}")
        
    async def on_tool_end(self, ctx, agent, tool, result):
        print(f"Tool {tool.name} completed")

agent = Agent(
    name="Monitored Agent",
    hooks=CustomHooks()
)
```

### MCP Integration

Connect to Model Context Protocol servers:

```python
from agents import Agent
from agents.mcp import MCPServer

# Connect to MCP server
mcp_server = MCPServer("stdio", "path/to/mcp/server")

agent = Agent(
    name="MCP Agent",
    mcp_servers=[mcp_server]
)
```

### Voice Integration

```python
from agents.voice import VoicePipeline, VoicePipelineConfig

config = VoicePipelineConfig(
    input_device="default",
    output_device="default",
    speech_model="whisper-1",
    tts_model="tts-1"
)

voice_agent = Agent(
    name="Voice Assistant",
    instructions="You are a voice assistant"
)

pipeline = VoicePipeline(voice_agent, config)
await pipeline.start()
```

---

## API Reference

### Core Classes

#### Agent
```python
class Agent[TContext]:
    name: str
    instructions: str | Callable
    model: str = "gpt-4o"
    model_settings: ModelSettings | None = None
    tools: list[Tool] = []
    handoffs: list[Agent | Handoff] = []
    guardrails: list[Guardrail] = []
    output_type: type = str
    hooks: AgentHooks | None = None
    mcp_servers: list[MCPServer] = []
```

#### Runner
```python
class Runner:
    @staticmethod
    async def run(
        agent: Agent,
        input: str | list,
        context: Any = None,
        session: Session | None = None,
        run_config: RunConfig | None = None
    ) -> RunResult
    
    @staticmethod
    def run_sync(
        agent: Agent,
        input: str | list,
        context: Any = None,
        session: Session | None = None,
        run_config: RunConfig | None = None
    ) -> RunResult
    
    @staticmethod
    async def run_streamed(
        agent: Agent,
        input: str | list,
        context: Any = None,
        session: Session | None = None,
        run_config: RunConfig | None = None
    ) -> RunResultStreaming
```

#### Tools
```python
# Hosted tools
WebSearchTool(max_results: int = 10)
FileSearchTool(vector_store_ids: list[str], max_num_results: int = 20)
ComputerTool()
CodeInterpreterTool()

# Function tool decorator
@function_tool(
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True
)
def your_function(...) -> ...:
    pass
```

### Configuration Classes

#### ModelSettings
```python
class ModelSettings:
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_completion_tokens: int | None = None
    extra_args: dict = {}
```

#### RunConfig
```python
class RunConfig:
    max_turns: int = 100
    model: str | None = None
    model_provider: ModelProvider | None = None
    tracing_disabled: bool = False
    trace_include_sensitive_data: bool = True
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Not Set
```
Error: OpenAI API key not provided
```
**Solution**: Set the `OPENAI_API_KEY` environment variable.

#### 2. Tool Call Failures
```
Error: Function 'my_tool' failed with error: ...
```
**Solution**: Add proper error handling in your function tools.

#### 3. Max Turns Exceeded
```
MaxTurnsExceeded: Agent exceeded maximum turns (100)
```
**Solution**: Increase `max_turns` in RunConfig or fix infinite loops.

#### 4. Guardrail Failures
```
InputGuardrailTripwireTriggered: Input blocked by guardrail
```
**Solution**: Handle the exception or adjust guardrail logic.

### Debugging Tips

1. **Enable Verbose Logging**:
   ```python
   import logging
   logging.getLogger("openai.agents").setLevel(logging.DEBUG)
   ```

2. **Use Tracing Dashboard**: Check traces at https://platform.openai.com/traces

3. **Test Tools Individually**:
   ```python
   # Test function tools directly
   result = await your_function_tool(test_input)
   ```

4. **Check Model Compatibility**: Ensure your model supports required features.

### Performance Optimization

1. **Use Appropriate Models**: Choose models based on task complexity
2. **Batch Operations**: Process multiple requests together when possible
3. **Cache Results**: Implement caching for expensive operations
4. **Optimize Prompts**: Keep instructions concise but clear
5. **Limit Tool Scope**: Only provide necessary tools to each agent

---

## Conclusion

This comprehensive manual covers all aspects of the OpenAI Agents SDK. For additional examples and advanced use cases, explore the [examples directory](https://github.com/NORA-BITSY/openai-agents-python/tree/main/examples) in the repository.

For the latest updates and detailed API documentation, visit the [official documentation](https://openai.github.io/openai-agents-python/).

The SDK's design philosophy emphasizes simplicity and flexibility - you can start with basic agents and gradually add sophisticated features like multi-agent handoffs, guardrails, and custom tools as your application grows.
