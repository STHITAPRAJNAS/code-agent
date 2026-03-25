# Google ADK Python Samples Research

Source: https://github.com/google/adk-python/tree/main/contributing/samples

---

## Directory Listing (134 samples total)

All sample names in `contributing/samples`:
a2a_auth, a2a_basic, a2a_human_in_loop, a2a_root, adk_answering_agent, adk_documentation, adk_issue_formatting_agent, adk_issue_monitoring_agent, adk_knowledge_agent, adk_pr_agent, adk_pr_triaging_agent, adk_stale_agent, adk_triaging_agent, agent_engine_code_execution, agent_registry_agent, api_registry_agent, application_integration_agent, artifact_save_text, authn-adk-all-in-one, bigquery, bigquery_mcp, bigtable, built_in_multi_tools, cache_analysis, callbacks, code_execution, computer_use, context_offloading_with_artifact, core_basic_config, core_callback_config, core_custom_agent_config, core_generate_content_config_config, crewai_tool_kwargs, custom_code_execution, data_agent, fields_output_schema, fields_planner, files_retrieval_agent, generate_image, gepa, gke_agent_sandbox, google_api, google_search_agent, hello_world, hello_world_anthropic, hello_world_apigeellm, hello_world_app, hello_world_gemma, hello_world_gemma3_ollama, hello_world_litellm, hello_world_litellm_add_function_to_prompt, hello_world_ma, hello_world_ollama, hello_world_stream_fc_args, history_management, human_in_loop, human_tool_confirmation, integration_connector_euc_agent, interactions_api, jira_agent, json_passing_agent, langchain_structured_tool_agent, langchain_youtube_search_agent, litellm_inline_tool_call, litellm_streaming, litellm_structured_output, litellm_with_fallback_models, live_agent_api_server_example, live_bidi_debug_utils, live_bidi_streaming_multi_agent, live_bidi_streaming_single_agent, live_bidi_streaming_tools_agent, live_tool_callbacks_agent, logprobs, manual_ollama_test, mcp_dynamic_header_agent, mcp_in_agent_tool_remote, mcp_in_agent_tool_stdio, mcp_postgres_agent, mcp_progress_callback_agent, mcp_server_side_sampling, mcp_service_account_agent, mcp_sse_agent, mcp_stdio_notion_agent, mcp_stdio_server_agent, mcp_streamablehttp_agent, mcp_toolset_auth, memory, migrate_session_db, multi_agent_basic_config, multi_agent_llm_config, multi_agent_loop_config, multi_agent_seq_config, multimodal_tool_results, non_llm_sequential, oauth2_client_credentials, oauth_calendar_agent, output_schema_with_tools, parallel_functions, plugin_basic, plugin_debug_logging, plugin_reflect_tool_retry, postgres_session_service, pubsub, pydantic_argument, quickstart, rag_agent, rewind_session, runner_debug_example, session_state_agent, simple_sequential_agent, skills_agent, skills_agent_gcs, slack_agent, spanner, spanner_admin, spanner_rag_agent, static_instruction, static_non_text_content, sub_agents_config, telemetry, token_usage, tool_agent_tool_config, tool_builtin_config, tool_functions_config, tool_human_in_the_loop_config, tool_mcp_stdio_notion_config, toolbox_agent, vertex_code_execution, workflow_agent_seq, workflow_triage

---

## 1. human_tool_confirmation

**File**: `contributing/samples/human_tool_confirmation/agent.py`

### Full Code

```python
from google.adk import Agent
from google.adk.apps import App
from google.adk.apps import ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def reimburse(amount: int, tool_context: ToolContext) -> str:
  """Reimburse the employee for the given amount."""
  return {'status': 'ok'}


async def confirmation_threshold(
    amount: int, tool_context: ToolContext
) -> bool:
  """Returns true if the amount is greater than 1000."""
  return amount > 1000


def request_time_off(days: int, tool_context: ToolContext):
  """Request day off for the employee."""
  if days <= 0:
    return {'status': 'Invalid days to request.'}

  if days <= 2:
    return {
        'status': 'ok',
        'approved_days': days,
    }

  tool_confirmation = tool_context.tool_confirmation
  if not tool_confirmation:
    tool_context.request_confirmation(
        hint=(
            'Please approve or reject the tool call request_time_off() by'
            ' responding with a FunctionResponse with an expected'
            ' ToolConfirmation payload.'
        ),
        payload={
            'approved_days': 0,
        },
    )
    return {'status': 'Manager approval is required.'}

  approved_days = tool_confirmation.payload['approved_days']
  approved_days = min(approved_days, days)
  if approved_days == 0:
    return {'status': 'The time off request is rejected.', 'approved_days': 0}
  return {
      'status': 'ok',
      'approved_days': approved_days,
  }


root_agent = Agent(
    model='gemini-2.5-flash',
    name='time_off_agent',
    instruction="""
    You are a helpful assistant that can help employees with reimbursement and time off requests.
    - Use the `reimburse` tool for reimbursement requests.
    - Use the `request_time_off` tool for time off requests.
    - Prioritize using tools to fulfill the user's request.
    - Always respond to the user with the tool results.
    """,
    tools=[
        FunctionTool(
            reimburse,
            require_confirmation=confirmation_threshold,
        ),
        request_time_off,
    ],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)

app = App(
    name='human_tool_confirmation',
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(
        is_resumable=True,
    ),
)
```

### Analysis

**Pattern**: Human-in-the-loop tool confirmation. Two distinct patterns demonstrated:

**Pattern A — `FunctionTool(require_confirmation=...)`** (simpler, boolean):
- Wrap the function in `FunctionTool` and pass `require_confirmation` as either:
  - `True` (always require confirmation)
  - A callable `async def confirmation_threshold(amount, tool_context) -> bool` that decides at runtime whether confirmation is needed
- The ADK framework pauses execution and waits for user confirmation before calling the actual function
- The callable receives the same args as the tool function plus `tool_context`

**Pattern B — `tool_context.request_confirmation(hint, payload)`** (richer, custom payload):
- Inside the tool function itself, check `tool_context.tool_confirmation`
- If it is `None` (first call, no confirmation yet), call `tool_context.request_confirmation(hint=..., payload=...)` and return early
- The `hint` string instructs the human what to respond with
- The `payload` is the default/initial data the human can modify
- On the resumed call, `tool_context.tool_confirmation` will be a `ToolConfirmation` object with `payload` set to what the human sent back
- The tool reads `tool_context.tool_confirmation.payload` to get the human's response

**Key ADK APIs**:
- `from google.adk.tools.function_tool import FunctionTool` — wraps a plain function, supports `require_confirmation`
- `from google.adk.tools.tool_confirmation import ToolConfirmation` — the object that carries the human's confirmation response
- `from google.adk.tools.tool_context import ToolContext` — provides `tool_context.tool_confirmation`, `tool_context.request_confirmation(hint, payload)`, `tool_context.state`
- `from google.adk.apps import App, ResumabilityConfig` — wraps agent in an App with resumability support
- `ResumabilityConfig(is_resumable=True)` — enables the agent to be paused and resumed

**Flow**:
1. Agent calls `request_time_off(days=5)`
2. Tool checks `tool_context.tool_confirmation` → `None`
3. Tool calls `tool_context.request_confirmation(hint="...", payload={"approved_days": 0})`
4. Tool returns `{'status': 'Manager approval is required.'}`
5. Execution pauses, human is shown the hint
6. Human responds with a `FunctionResponse` containing `ToolConfirmation(payload={"approved_days": 3})`
7. Agent is resumed; ADK calls `request_time_off(days=5)` again
8. This time `tool_context.tool_confirmation.payload == {"approved_days": 3}`
9. Tool uses the approved value and returns final result

---

## 2. human_in_loop (LongRunningFunctionTool)

**File**: `contributing/samples/human_in_loop/agent.py`

### Full Code

```python
from typing import Any

from google.adk import Agent
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def reimburse(purpose: str, amount: float) -> str:
  """Reimburse the amount of money to the employee."""
  return {
      'status': 'ok',
  }


def ask_for_approval(
    purpose: str, amount: float, tool_context: ToolContext
) -> dict[str, Any]:
  """Ask for approval for the reimbursement."""
  return {
      'status': 'pending',
      'amount': amount,
      'ticketId': 'reimbursement-ticket-001',
  }


root_agent = Agent(
    model='gemini-2.5-flash',
    name='reimbursement_agent',
    instruction="""
      You are an agent whose job is to handle the reimbursement process for
      the employees. If the amount is less than $100, you will automatically
      approve the reimbursement.

      If the amount is greater than $100, you will
      ask for approval from the manager. If the manager approves, you will
      call reimburse() to reimburse the amount to the employee. If the manager
      rejects, you will inform the employee of the rejection.
""",
    tools=[reimburse, LongRunningFunctionTool(func=ask_for_approval)],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)
```

### Analysis

**Pattern**: Long-running tool that returns an intermediate status and waits for an external event (human approval, webhook, etc.) before the agent continues.

**How `LongRunningFunctionTool` works**:
- Import: `from google.adk.tools.long_running_tool import LongRunningFunctionTool`
- Wrap a function: `LongRunningFunctionTool(func=ask_for_approval)`
- The wrapped function runs immediately and returns a dict with `status: 'pending'` (or any initial response)
- The ADK framework yields an event indicating the tool result is pending
- The agent does NOT terminate — it waits for an external resume signal
- An external system (human, webhook, etc.) sends a `FunctionResponse` with the ticket ID and final result
- The agent resumes and processes the response as the tool result

**Resume mechanism**:
- The initial tool response contains `ticketId` which acts as a correlation ID
- External system calls back with the ticket ID and status (`approved`/`rejected`)
- ADK matches the response to the pending tool call and continues the agent run

**Key ADK APIs**:
- `LongRunningFunctionTool(func=...)` — the main wrapper
- Returns any dict immediately (the "pending" receipt)
- External resume is handled at the App/Runner level by sending a `FunctionResponse` back into the session

---

## 3. callbacks

**File**: `contributing/samples/callbacks/agent.py`

### Full Code

```python
import random

from google.adk import Agent
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def roll_die(sides: int, tool_context: ToolContext) -> int:
  """Roll a die and return the rolled result."""
  result = random.randint(1, sides)
  if not 'rolls' in tool_context.state:
    tool_context.state['rolls'] = []
  tool_context.state['rolls'] = tool_context.state['rolls'] + [result]
  return result


async def check_prime(nums: list[int]) -> str:
  """Check if a given list of numbers are prime."""
  primes = set()
  for number in nums:
    number = int(number)
    if number <= 1:
      continue
    is_prime = True
    for i in range(2, int(number**0.5) + 1):
      if number % i == 0:
        is_prime = False
        break
    if is_prime:
      primes.add(number)
  return (
      'No prime numbers found.'
      if not primes
      else f"{', '.join(str(num) for num in primes)} are prime numbers."
  )


async def before_agent_callback(callback_context):
  print('@before_agent_callback')
  return None


async def after_agent_callback(callback_context):
  print('@after_agent_callback')
  return None


async def before_model_callback(callback_context, llm_request):
  print('@before_model_callback')
  return None


async def after_model_callback(callback_context, llm_response):
  print('@after_model_callback')
  return None


def after_agent_cb1(callback_context):
  print('@after_agent_cb1')


def after_agent_cb2(callback_context):
  print('@after_agent_cb2')
  # Return ModelContent to short-circuit — agent stops processing
  return types.ModelContent(
      parts=[
          types.Part(
              text='(stopped) after_agent_cb2',
          ),
      ],
  )


def after_agent_cb3(callback_context):
  print('@after_agent_cb3')


def before_agent_cb1(callback_context):
  print('@before_agent_cb1')


def before_agent_cb2(callback_context):
  print('@before_agent_cb2')


def before_agent_cb3(callback_context):
  print('@before_agent_cb3')


def before_tool_cb1(tool, args, tool_context):
  print('@before_tool_cb1')


def before_tool_cb2(tool, args, tool_context):
  print('@before_tool_cb2')


def before_tool_cb3(tool, args, tool_context):
  print('@before_tool_cb3')


def after_tool_cb1(tool, args, tool_context, tool_response):
  print('@after_tool_cb1')


def after_tool_cb2(tool, args, tool_context, tool_response):
  print('@after_tool_cb2')
  # Return a dict to override the tool response
  return {'test': 'after_tool_cb2', 'response': tool_response}


def after_tool_cb3(tool, args, tool_context, tool_response):
  print('@after_tool_cb3')


root_agent = Agent(
    model='gemini-2.0-flash',
    name='data_processing_agent',
    description='hello world agent that can roll a dice of 8 sides and check prime numbers.',
    instruction="""...""",
    tools=[roll_die, check_prime],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
    before_agent_callback=[before_agent_cb1, before_agent_cb2, before_agent_cb3],
    after_agent_callback=[after_agent_cb1, after_agent_cb2, after_agent_cb3],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=[before_tool_cb1, before_tool_cb2, before_tool_cb3],
    after_tool_callback=[after_tool_cb1, after_tool_cb2, after_tool_cb3],
)
```

### Callback Signatures

| Callback | Signature | Return to override |
|---|---|---|
| `before_agent_callback` | `(callback_context) -> Optional[types.Content]` | Return `types.ModelContent(...)` to skip agent |
| `after_agent_callback` | `(callback_context) -> Optional[types.Content]` | Return `types.ModelContent(...)` to replace output |
| `before_model_callback` | `(callback_context, llm_request: LlmRequest) -> None` | Return `LlmResponse` to skip model call |
| `after_model_callback` | `(callback_context, llm_response: LlmResponse) -> None` | Return new `LlmResponse` to replace |
| `before_tool_callback` | `(tool, args, tool_context) -> None` | Return dict to skip tool execution |
| `after_tool_callback` | `(tool, args, tool_context, tool_response) -> None` | Return dict to replace tool response |

**Key points**:
- All callbacks can be a single callable OR a list of callables
- Can be sync or async
- `before_agent_callback` returning `types.ModelContent(...)` will short-circuit the entire agent turn
- `after_tool_callback` returning a dict overrides what the model sees as the tool response
- `callback_context` is a `CallbackContext` with `.state` dict (session state), `._invocation_context`
- `tool_context` is a `ToolContext` with `.state` dict

---

## 4. simple_sequential_agent

**File**: `contributing/samples/simple_sequential_agent/agent.py`

### Full Code

```python
import random

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.genai import types


def roll_die(sides: int) -> int:
  """Roll a die and return the rolled result."""
  return random.randint(1, sides)


roll_agent = LlmAgent(
    name="roll_agent",
    description="Handles rolling dice of different sizes.",
    model="gemini-2.0-flash",
    instruction="""
      You are responsible for rolling dice based on the user's request.
      When asked to roll a die, you must call the roll_die tool with the number of sides as an integer.
    """,
    tools=[roll_die],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)


def check_prime(nums: list[int]) -> str:
  """Check if a given list of numbers are prime."""
  primes = set()
  for number in nums:
    number = int(number)
    if number <= 1:
      continue
    is_prime = True
    for i in range(2, int(number**0.5) + 1):
      if number % i == 0:
        is_prime = False
        break
    if is_prime:
      primes.add(number)
  return (
      "No prime numbers found."
      if not primes
      else f"{', '.join(str(num) for num in primes)} are prime numbers."
  )


prime_agent = LlmAgent(
    name="prime_agent",
    description="Handles checking if numbers are prime.",
    model="gemini-2.0-flash",
    instruction="""
      You are responsible for checking whether numbers are prime.
      When asked to check primes, you must call the check_prime tool with a list of integers.
      Never attempt to determine prime numbers manually.
      Return the prime number results to the root agent.
    """,
    tools=[check_prime],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)

root_agent = SequentialAgent(
    name="simple_sequential_agent",
    sub_agents=[roll_agent, prime_agent],
    # Agents run in order: roll_agent -> prime_agent
)
```

### Analysis

**Pattern**: Sequential multi-agent pipeline. Sub-agents run one after another in list order. Each sub-agent has full LLM capabilities. The output of one becomes available (via shared session state or conversation history) for the next.

**Key ADK APIs**:
- `from google.adk.agents.sequential_agent import SequentialAgent`
- `from google.adk.agents.llm_agent import LlmAgent` (alias: `Agent`)
- `SequentialAgent(name=..., sub_agents=[agent1, agent2, ...])`
- Sub-agents communicate via session state (`output_key` pattern) or conversation history

---

## 5. workflow_agent_seq (Advanced Sequential — Code Pipeline)

**File**: `contributing/samples/workflow_agent_seq/agent.py`

### Full Code

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent


code_writer_agent = LlmAgent(
    name="CodeWriterAgent",
    model="gemini-2.5-flash",
    instruction="""You are a Python Code Generator.
Based *only* on the user's request, write Python code that fulfills the requirement.
Output *only* the complete Python code block, enclosed in triple backticks (```python ... ```).
Do not add any other text before or after the code block.
""",
    description="Writes initial Python code based on a specification.",
    output_key="generated_code",  # Stores output in state['generated_code']
)

code_reviewer_agent = LlmAgent(
    name="CodeReviewerAgent",
    model="gemini-2.5-flash",
    instruction="""You are an expert Python Code Reviewer.
    Your task is to provide constructive feedback on the provided code.

    **Code to Review:**
    ```python
    {generated_code}
    ```
...""",
    description="Reviews code and provides feedback.",
    output_key="review_comments",  # Stores output in state['review_comments']
)

code_refactorer_agent = LlmAgent(
    name="CodeRefactorerAgent",
    model="gemini-2.5-flash",
    instruction="""You are a Python Code Refactoring AI.
Your goal is to improve the given Python code based on the provided review comments.

  **Original Code:**
  ```python
  {generated_code}
  ```

  **Review Comments:**
  {review_comments}
...""",
    description="Refactors code based on review comments.",
    output_key="refactored_code",
)

code_pipeline_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[code_writer_agent, code_reviewer_agent, code_refactorer_agent],
    description="Executes a sequence of code writing, reviewing, and refactoring.",
)

root_agent = code_pipeline_agent
```

### Analysis

**Key pattern**: `output_key="state_key_name"` on an LlmAgent causes its output to be written to `session.state["state_key_name"]`. Subsequent agents can reference it in their instructions via `{state_key_name}` template substitution. This is the primary way sequential agents pass data to each other.

---

## 6. parallel_functions

**File**: `contributing/samples/parallel_functions/agent.py`

### Full Code (abbreviated to key structure)

```python
import asyncio
import time
from typing import List

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext


async def get_weather(city: str, tool_context: ToolContext) -> dict:
  """Get the current weather for a city."""
  await asyncio.sleep(2)  # Simulates async work
  # ... mock data ...
  if 'weather_requests' not in tool_context.state:
    tool_context.state['weather_requests'] = []
  tool_context.state['weather_requests'].append(
      {'city': city, 'timestamp': time.time(), 'result': result}
  )
  return {...}


async def get_currency_rate(from_currency: str, to_currency: str, tool_context: ToolContext) -> dict:
  await asyncio.sleep(1.5)
  if 'currency_requests' not in tool_context.state:
    tool_context.state['currency_requests'] = []
  tool_context.state['currency_requests'].append({...})
  return {...}


async def calculate_distance(city1: str, city2: str, tool_context: ToolContext) -> dict:
  await asyncio.sleep(1)
  if 'distance_requests' not in tool_context.state:
    tool_context.state['distance_requests'] = []
  tool_context.state['distance_requests'].append({...})
  return {...}


async def get_population(cities: List[str], tool_context: ToolContext) -> dict:
  await asyncio.sleep(len(cities) * 0.5)
  if 'population_requests' not in tool_context.state:
    tool_context.state['population_requests'] = []
  tool_context.state['population_requests'].append({...})
  return {...}


root_agent = Agent(
    model='gemini-2.0-flash',
    name='parallel_function_test_agent',
    description='Agent for testing parallel function calling performance and thread safety.',
    instruction="""
    When users ask for information about multiple cities or multiple types of data,
    you should call multiple functions in parallel to provide faster responses.
    """,
    tools=[get_weather, get_currency_rate, calculate_distance, get_population],
)
```

### Analysis

**Pattern**: Parallel function calling. The LLM is instructed to call multiple tools simultaneously. The ADK executes them concurrently.

**Key patterns**:
- Tools are `async` functions — allows true async concurrency
- `tool_context.state` can be written to from multiple concurrent tool calls (thread-safety note in code)
- State mutation pattern: always read, mutate, write (not in-place mutation of mutable objects in state): `tool_context.state['key'] = tool_context.state['key'] + [new_item]`
- The agent instruction explicitly tells the model to call functions in parallel

---

## 7. hello_world_litellm (LiteLlm)

**File**: `contributing/samples/hello_world_litellm/agent.py`

### Full Code

```python
import random

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm


def roll_die(sides: int) -> int:
  """Roll a die and return the rolled result."""
  return random.randint(1, sides)


async def check_prime(nums: list[int]) -> str:
  """Check if a given list of numbers are prime."""
  # ... implementation ...


root_agent = Agent(
    # model=LiteLlm(model="gemini/gemini-2.5-pro-exp-03-25"),
    # model=LiteLlm(model="vertex_ai/gemini-2.5-pro-exp-03-25"),
    # model=LiteLlm(model="vertex_ai/claude-3-5-haiku"),
    model=LiteLlm(model="openai/gpt-4o"),
    # model=LiteLlm(model="anthropic/claude-3-sonnet-20240229"),
    name="data_processing_agent",
    description="hello world agent that can roll a dice of 8 sides and check prime numbers.",
    instruction="""...""",
    tools=[roll_die, check_prime],
)
```

### Analysis

**Pattern**: Drop-in replacement of the model parameter using LiteLLM to support any provider.

**LiteLlm configuration**:
- `from google.adk.models.lite_llm import LiteLlm`
- `model=LiteLlm(model="<provider>/<model_name>")`
- Supported provider prefixes: `openai/`, `anthropic/`, `gemini/`, `vertex_ai/`, and all other LiteLLM providers
- The rest of the agent definition is unchanged

---

## 8. litellm_streaming

**File**: `contributing/samples/litellm_streaming/agent.py`

```python
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    name='litellm_streaming_agent',
    model=LiteLlm(model='gemini/gemini-2.5-flash'),
    description='A LiteLLM agent used for streaming text responses.',
    instruction='You are a verbose assistant',
)
```

---

## 9. litellm_structured_output

**File**: `contributing/samples/litellm_structured_output/agent.py`

```python
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field


class CitySummary(BaseModel):
  city: str = Field(description="Name of the city being described.")
  highlights: list[str] = Field(
      description="Bullet points summarising the city's key highlights.",
  )
  recommended_visit_length_days: int = Field(
      description="Recommended number of days for a typical visit.",
  )


root_agent = Agent(
    name="litellm_structured_output_agent",
    model=LiteLlm(model="gemini-2.5-flash"),
    description="Generates structured travel recommendations for a given city.",
    instruction="""
Produce a JSON object that follows the CitySummary schema.
Only include fields that appear in the schema and ensure highlights
contains short bullet points.
""".strip(),
    output_schema=CitySummary,
)
```

---

## 10. litellm_with_fallback_models

**File**: `contributing/samples/litellm_with_fallback_models/agent.py`

```python
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def roll_die(sides: int, tool_context: ToolContext) -> int:
  result = random.randint(1, sides)
  if 'rolls' not in tool_context.state:
    tool_context.state['rolls'] = []
  tool_context.state['rolls'] = tool_context.state['rolls'] + [result]
  return result


async def before_model_callback(callback_context, llm_request):
  print(f'Beginning model choice: {llm_request.model}')
  callback_context.state['beginning_model_choice'] = llm_request.model
  return None


async def after_model_callback(callback_context, llm_response):
  print(f'Final model choice: {llm_response.model_version}')
  callback_context.state['final_model_choice'] = llm_response.model_version
  return None


root_agent = Agent(
    model=LiteLlm(
        model='gemini/gemini-2.5-pro',
        fallbacks=[
            'anthropic/claude-sonnet-4-5-20250929',
            'openai/gpt-4o',
        ],
    ),
    name='resilient_agent',
    ...
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)
```

### Analysis

**LiteLlm fallback configuration**:
- `LiteLlm(model='primary/model', fallbacks=['provider2/model2', 'provider3/model3'])`
- If primary fails, LiteLLM automatically tries each fallback in order
- `before_model_callback` has access to `llm_request.model` — the currently chosen model
- `after_model_callback` has access to `llm_response.model_version` — the model that actually responded

---

## 11. litellm_inline_tool_call (Custom LiteLLMClient)

**File**: `contributing/samples/litellm_inline_tool_call/agent.py`

```python
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.lite_llm import LiteLLMClient


class InlineJsonToolClient(LiteLLMClient):
  """LiteLLM client that emits inline JSON tool calls for testing."""

  async def acompletion(self, model, messages, tools, **kwargs):
    del tools, kwargs
    tool_message = _find_last_role(messages, role="tool")
    if tool_message:
      tool_summary = _coerce_to_text(tool_message.get("content"))
      return {
          "id": "mock-inline-tool-final-response",
          "model": model,
          "choices": [{"message": {"role": "assistant", "content": f"The instrumentation tool responded with: {tool_summary}"}, "finish_reason": "stop"}],
          "usage": {"prompt_tokens": 60, "completion_tokens": 12, "total_tokens": 72},
      }
    timezone = _extract_timezone(messages) or "Asia/Taipei"
    inline_call = json.dumps({"name": "get_current_time", "arguments": {"timezone_str": timezone}}, separators=(",", ":"))
    return {
        "id": "mock-inline-tool-call",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": f"{inline_call}\nLet me double-check the clock for you."}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 45, "completion_tokens": 15, "total_tokens": 60},
    }


_mock_model = LiteLlm(
    model="mock/inline-json-tool-calls",
    llm_client=InlineJsonToolClient(),
)

root_agent = Agent(
    name="litellm_inline_tool_tester",
    model=_mock_model,
    ...
    tools=[get_current_time],
)
```

### Analysis

**Pattern**: Custom LiteLLM backend by subclassing `LiteLLMClient` and overriding `async def acompletion(self, model, messages, tools, **kwargs)`.

**Key API**: `LiteLlm(model="...", llm_client=CustomLiteLLMClient())` — inject any custom client that implements the OpenAI completion response format.

---

## 12. hello_world_litellm_add_function_to_prompt (LiteLLM with LangChain function format)

**File**: `contributing/samples/hello_world_litellm_add_function_to_prompt/agent.py`

```python
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from langchain_core.utils.function_calling import convert_to_openai_function


root_agent = Agent(
    model=LiteLlm(
        model="vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas",
        functions=[
            convert_to_openai_function(roll_die),
            convert_to_openai_function(check_prime),
        ],
    ),
    name="data_processing_agent",
    ...
    tools=[roll_die, check_prime],
)
```

### Analysis

**Pattern**: For models that don't natively support tool schemas via LiteLLM, pass OpenAI-format function definitions directly via `LiteLlm(functions=[...])`. The `convert_to_openai_function` utility from LangChain converts Python functions to OpenAI function calling format.

---

## 13. hello_world_anthropic (Claude via native ADK)

**File**: `contributing/samples/hello_world_anthropic/agent.py`

```python
from google.adk import Agent
from google.adk.models.anthropic_llm import Claude


root_agent = Agent(
    model=Claude(model="claude-3-5-sonnet-v2@20241022"),
    name="hello_world_agent",
    ...
    tools=[roll_die, check_prime],
)
```

### Analysis

**Pattern**: Native Anthropic Claude support via `from google.adk.models.anthropic_llm import Claude`. Use `Claude(model="<claude-model-id>")` as the model parameter.

---

## 14. session_state_agent

**File**: `contributing/samples/session_state_agent/agent.py`

### Full Code

```python
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


async def assert_session_values(
    ctx: CallbackContext,
    title: str,
    *,
    keys_in_ctx_session: Optional[list[str]] = None,
    keys_in_service_session: Optional[list[str]] = None,
    keys_not_in_service_session: Optional[list[str]] = None,
):
  session_in_ctx = ctx._invocation_context.session
  session_in_service = (
      await ctx._invocation_context.session_service.get_session(
          app_name=session_in_ctx.app_name,
          user_id=session_in_ctx.user_id,
          session_id=session_in_ctx.id,
      )
  )
  # Assertion checks...


async def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
  if 'before_agent_callback_state_key' in callback_context.state:
    return types.ModelContent('Sorry, I can only reply once.')
  callback_context.state['before_agent_callback_state_key'] = 'before_agent_callback_state_value'


async def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest):
  callback_context.state['before_model_callback_state_key'] = 'before_model_callback_state_value'


async def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse):
  callback_context.state['after_model_callback_state_key'] = 'after_model_callback_state_value'


async def after_agent_callback(callback_context: CallbackContext):
  callback_context.state['after_agent_callback_state_key'] = 'after_agent_callback_state_value'


root_agent = Agent(
    name='root_agent',
    description='a verification agent.',
    instruction=(
        'Log all users query with `log_query` tool. Must always remind user you'
        ' cannot answer second query because your setup.'
    ),
    model='gemini-2.5-flash',
    before_agent_callback=before_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)
```

### Analysis

**Session state lifecycle** (critical insight from this sample):

State changes are **buffered in memory** during a turn and **persisted to the backing store at specific synchronization points**:

| Callback | Keys that ARE persisted in service | Keys that are NOT yet persisted |
|---|---|---|
| `before_agent_callback` | (nothing from this turn) | `before_agent_callback_state_key` |
| `before_model_callback` | `before_agent_callback_state_key` | `before_model_callback_state_key` |
| `after_model_callback` | `before_agent_callback_state_key` | `before_model_callback_state_key`, `after_model_callback_state_key` |
| `after_agent_callback` | `before_agent_callback_state_key`, `before_model_callback_state_key`, `after_model_callback_state_key` | `after_agent_callback_state_key` |

**Rules**:
- `callback_context.state` always has the latest in-memory values
- The session service only persists a snapshot at certain checkpoints (after agent start, after model call)
- `after_agent_callback` state changes are persisted after the callback completes

**Access patterns**:
- `callback_context.state['key'] = value` — write to session state
- `callback_context.state.get('key')` or `'key' in callback_context.state` — read
- `ctx._invocation_context.session` — the in-memory Session object
- `ctx._invocation_context.session_service.get_session(app_name, user_id, session_id)` — fetch from backing store (async)

---

## 15. rewind_session (session state + artifacts)

**File**: `contributing/samples/rewind_session/agent.py`

```python
from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types


async def update_state(tool_context: ToolContext, key: str, value: str) -> dict:
  """Updates a state value."""
  tool_context.state[key] = value
  return {"status": f"Updated state '{key}' to '{value}'"}


async def load_state(tool_context: ToolContext, key: str) -> dict:
  """Loads a state value."""
  return {key: tool_context.state.get(key)}


async def save_artifact(
    tool_context: ToolContext, filename: str, content: str
) -> dict:
  """Saves an artifact with the given filename and content."""
  artifact_bytes = content.encode("utf-8")
  artifact_part = types.Part(
      inline_data=types.Blob(mime_type="text/plain", data=artifact_bytes)
  )
  version = await tool_context.save_artifact(filename, artifact_part)
  return {"status": "success", "filename": filename, "version": version}


async def load_artifact(tool_context: ToolContext, filename: str) -> dict:
  """Loads an artifact with the given filename."""
  artifact = await tool_context.load_artifact(filename)
  if not artifact:
    return {"error": f"Artifact '{filename}' not found"}
  content = artifact.inline_data.data.decode("utf-8")
  return {"filename": filename, "content": content}


root_agent = Agent(
    name="state_agent",
    model="gemini-2.0-flash",
    instruction="""You are an agent that manages state and artifacts.
    You can: Update state value, Load state value, Save artifact, Load artifact.""",
    tools=[update_state, load_state, save_artifact, load_artifact],
)
```

### Analysis

**Tool Context APIs for state and artifacts**:
- `tool_context.state[key] = value` — write session state
- `tool_context.state.get(key)` — read session state
- `await tool_context.save_artifact(filename, types.Part(...))` — save an artifact, returns version number
- `await tool_context.load_artifact(filename)` — load an artifact, returns `types.Part` or `None`
- Artifacts use `types.Part(inline_data=types.Blob(mime_type="...", data=bytes))`

---

## 16. history_management

**File**: `contributing/samples/history_management/agent.py`

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest


def create_slice_history_callback(n_recent_turns):
  async def before_model_callback(
      callback_context: CallbackContext, llm_request: LlmRequest
  ):
    if n_recent_turns < 1:
      return

    user_indexes = [
        i
        for i, content in enumerate(llm_request.contents)
        if content.role == 'user'
    ]

    if n_recent_turns > len(user_indexes):
      return

    suffix_idx = user_indexes[-n_recent_turns]
    llm_request.contents = llm_request.contents[suffix_idx:]

  return before_model_callback


root_agent = Agent(
    model='gemini-2.0-flash',
    name='short_history_agent',
    ...
    before_model_callback=create_slice_history_callback(n_recent_turns=2),
)
```

### Analysis

**Pattern**: Truncate conversation history before each model call. The `before_model_callback` receives `llm_request: LlmRequest` and can mutate `llm_request.contents` to control what history the model sees.

**Key APIs**:
- `llm_request.contents` — list of `Content` objects (conversation history)
- `content.role` — `'user'` or `'model'`
- Mutate `llm_request.contents` in-place before the model call
- Factory function pattern for parameterized callbacks

---

## 17. memory

**File**: `contributing/samples/memory/agent.py`

```python
from datetime import datetime

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.load_memory_tool import load_memory_tool
from google.adk.tools.preload_memory_tool import preload_memory_tool


def update_current_time(callback_context: CallbackContext):
  callback_context.state['_time'] = datetime.now().isoformat()


root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='memory_agent',
    description='agent that have access to memory tools.',
    before_agent_callback=update_current_time,
    instruction="""\
You are an agent that help user answer questions.

Current time: {_time}
""",
    tools=[
        load_memory_tool,
        preload_memory_tool,
    ],
)
```

### Analysis

**Pattern**: Memory-enabled agent. ADK provides built-in memory tools.

**Key APIs**:
- `from google.adk.tools.load_memory_tool import load_memory_tool` — tool that loads memories into context
- `from google.adk.tools.preload_memory_tool import preload_memory_tool` — preloads memories before agent starts
- State keys prefixed with `_` (e.g., `_time`) are used in instruction templates via `{_time}`
- Sync callback — `before_agent_callback` can be sync

---

## 18. A2A Samples

### a2a_basic (RemoteA2aAgent)

**File**: `contributing/samples/a2a_basic/agent.py`

```python
import random

from google.adk.agents.llm_agent import Agent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.example_tool import ExampleTool
from google.genai import types


roll_agent = Agent(name="roll_agent", ...)

example_tool = ExampleTool([
    {
        "input": {"role": "user", "parts": [{"text": "Roll a 6-sided die."}]},
        "output": [{"role": "model", "parts": [{"text": "I rolled a 4 for you."}]}],
    },
    ...
])

prime_agent = RemoteA2aAgent(
    name="prime_agent",
    description="Agent that handles checking if numbers are prime.",
    agent_card=(
        f"http://localhost:8001/a2a/check_prime_agent{AGENT_CARD_WELL_KNOWN_PATH}"
    ),
)

root_agent = Agent(
    model="gemini-2.0-flash",
    name="root_agent",
    instruction="""...""",
    sub_agents=[roll_agent, prime_agent],
    tools=[example_tool],
    ...
)
```

**Key APIs**:
- `from google.adk.agents.remote_a2a_agent import RemoteA2aAgent` — proxies to a remote A2A agent
- `from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH` — constant for the `/.well-known/agent.json` path
- `RemoteA2aAgent(name=..., description=..., agent_card="http://host/a2a/agent_name/.well-known/agent.json")`
- `from google.adk.tools.example_tool import ExampleTool` — provides few-shot examples to guide the model

### a2a_human_in_loop (A2A with LongRunning remote approval)

**File**: `contributing/samples/a2a_human_in_loop/agent.py`

```python
from google.adk.agents.llm_agent import Agent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.genai import types


def reimburse(purpose: str, amount: float) -> str:
  return {'status': 'ok'}


approval_agent = RemoteA2aAgent(
    name='approval_agent',
    description='Help approve the reimburse if the amount is greater than 100.',
    agent_card=(
        f'http://localhost:8001/a2a/human_in_loop{AGENT_CARD_WELL_KNOWN_PATH}'
    ),
)

root_agent = Agent(
    model='gemini-2.0-flash',
    name='reimbursement_agent',
    instruction="""
      You are an agent whose job is to handle the reimbursement process for
      the employees. If the amount is less than $100, you will automatically
      approve the reimbursement. And call reimburse() to reimburse the amount to the employee.

      If the amount is greater than $100. You will hand over the request to
      approval_agent to handle the reimburse.
""",
    tools=[reimburse],
    sub_agents=[approval_agent],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)
```

**Pattern**: The human-in-loop approval is handled by a remote A2A agent running separately. The root agent delegates to the remote approval agent which implements the long-running confirmation flow.

### a2a_root (proxy to remote)

```python
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

root_agent = RemoteA2aAgent(
    name="hello_world_agent",
    description="Helpful assistant that can roll dice and check if numbers are prime.",
    agent_card=f"http://localhost:8001/{AGENT_CARD_WELL_KNOWN_PATH}",
)
```

**Pattern**: The entire root agent is a remote proxy. The root_agent variable can be a `RemoteA2aAgent` directly.

---

## 19. non_llm_sequential (SequentialAgent with minimal LLM agents)

```python
from google.adk.agents.llm_agent import Agent
from google.adk.agents.sequential_agent import SequentialAgent

sub_agent_1 = Agent(name='sub_agent_1', model='gemini-2.0-flash-001', instruction='JUST SAY 1.')
sub_agent_2 = Agent(name='sub_agent_2', model='gemini-2.0-flash-001', instruction='JUST SAY 2.')

sequential_agent = SequentialAgent(
    name='sequential_agent',
    sub_agents=[sub_agent_1, sub_agent_2],
)

root_agent = sequential_agent
```

---

## 20. Plugin System

### plugin_basic/count_plugin.py

```python
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin


class CountInvocationPlugin(BasePlugin):
  """A custom plugin that counts agent and tool invocations."""

  def __init__(self) -> None:
    super().__init__(name="count_invocation")
    self.agent_count: int = 0
    self.tool_count: int = 0
    self.llm_request_count: int = 0

  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> None:
    self.agent_count += 1
    print(f"[Plugin] Agent run count: {self.agent_count}")

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    self.llm_request_count += 1
    print(f"[Plugin] LLM request count: {self.llm_request_count}")
```

### plugin_basic/main.py

```python
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from .count_plugin import CountInvocationPlugin


root_agent = Agent(
    model='gemini-2.0-flash',
    name='hello_world',
    ...
    tools=[hello_world],
)

async def main():
  runner = InMemoryRunner(
      agent=root_agent,
      app_name='test_app_with_plugin',
      plugins=[CountInvocationPlugin()],
  )
  session = await runner.session_service.create_session(
      user_id='user',
      app_name='test_app_with_plugin',
  )
  async for event in runner.run_async(
      user_id='user',
      session_id=session.id,
      new_message=types.Content(role='user', parts=[types.Part.from_text(text=prompt)]),
  ):
    print(f'** Got event from {event.author}')
```

### Analysis

**Plugin pattern**:
- Subclass `from google.adk.plugins.base_plugin import BasePlugin`
- Override lifecycle methods with **keyword-only args** (`*, agent: BaseAgent, callback_context: CallbackContext`)
- Plugin callbacks differ from agent callbacks in signature: plugins use `*` keyword-only args
- Register plugins in `InMemoryRunner(plugins=[MyPlugin()])`
- Plugins apply globally across all agents, unlike per-agent callbacks

**Plugin callback signatures** (keyword-only args, different from per-agent callbacks):
```python
async def before_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> None
async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> None
```

---

## 21. YAML Config-Based Agents

### tool_human_in_the_loop_config/root_agent.yaml

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
name: reimbursement_agent
model: gemini-2.0-flash
instruction: |
  You are an agent whose job to handle the reimbursement process for the employees.
  ...
tools:
  - name: tool_human_in_the_loop_config.tools.reimburse
  - name: LongRunningFunctionTool
    args:
      func: tool_human_in_the_loop_config.tools.ask_for_approval
```

### core_callback_config/root_agent.yaml

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
name: hello_world_agent
model: gemini-2.0-flash
description: hello world agent that can roll a dice and check prime numbers.
instruction: |
  ...
tools:
  - name: core_callback_config.tools.roll_die
  - name: core_callback_config.tools.check_prime
before_agent_callbacks:
  - name: core_callback_config.callbacks.before_agent_callback1
  - name: core_callback_config.callbacks.before_agent_callback2
  - name: core_callback_config.callbacks.before_agent_callback3
after_agent_callbacks:
  - name: core_callback_config.callbacks.after_agent_callback1
  - name: core_callback_config.callbacks.after_agent_callback2
  - name: core_callback_config.callbacks.after_agent_callback3
before_model_callbacks:
  - name: core_callback_config.callbacks.before_model_callback
after_model_callbacks:
  - name: core_callback_config.callbacks.after_model_callback
before_tool_callbacks:
  - name: core_callback_config.callbacks.before_tool_callback1
  - name: core_callback_config.callbacks.before_tool_callback2
  - name: core_callback_config.callbacks.before_tool_callback3
after_tool_callbacks:
  - name: core_callback_config.callbacks.after_tool_callback1
  - name: core_callback_config.callbacks.after_tool_callback2
  - name: core_callback_config.callbacks.after_tool_callback3
```

### Analysis

**YAML-based agent config**:
- JSON Schema: `AgentConfig.json` from the ADK repo
- Tools referenced by dotted Python import path: `module.submodule.function_name`
- Special tool wrappers like `LongRunningFunctionTool` referenced by class name with `args.func` as import path
- Callbacks listed under `before_agent_callbacks` (plural), `after_agent_callbacks`, etc.
- All tools and callbacks resolved by import path at load time

---

## 22. fields_planner (BuiltInPlanner / PlanReActPlanner)

**File**: `contributing/samples/fields_planner/agent.py`

```python
from google.adk.agents.llm_agent import Agent
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.adk.tools.tool_context import ToolContext
from google.genai import types


root_agent = Agent(
    model='gemini-2.5-pro-preview-03-25',
    name='data_processing_agent',
    ...
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
        ),
    ),
    # planner=PlanReActPlanner(),
)
```

**Planner APIs**:
- `from google.adk.planners.built_in_planner import BuiltInPlanner` — native Gemini thinking/planning
- `from google.adk.planners.plan_re_act_planner import PlanReActPlanner` — ReAct-style planning
- `BuiltInPlanner(thinking_config=types.ThinkingConfig(include_thoughts=True))` — expose chain-of-thought

---

## 23. fields_output_schema

```python
from google.adk import Agent
from pydantic import BaseModel


class WeatherData(BaseModel):
  temperature: str
  humidity: str
  wind_speed: str


root_agent = Agent(
    name='root_agent',
    model='gemini-2.5-flash',
    instruction="""...""",
    output_schema=list[WeatherData],
    output_key='weather_data',
    tools=[get_current_year],
)
```

**Key APIs**:
- `output_schema=SomePydanticModel` or `output_schema=list[SomePydanticModel]` — agent output must conform to this schema
- `output_key='key_name'` — stores agent output in `session.state['key_name']` (used in sequential pipelines)

---

## 24. MCP Integration Samples

### mcp_in_agent_tool_stdio

```python
from google.adk.agents import Agent
from google.adk.tools import AgentTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters


mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uvx",
            args=["--from", "git+...", "mcp-simple-tool"],
        ),
        timeout=10.0,
    )
)

sub_agent = Agent(name="mcp_helper", model="gemini-2.5-flash", tools=[mcp_toolset])
mcp_agent_tool = AgentTool(agent=sub_agent)

root_agent = Agent(name="main_agent", model="gemini-2.5-flash", tools=[mcp_agent_tool])
```

### mcp_in_agent_tool_remote (SSE)

```python
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

mcp_toolset = McpToolset(
    connection_params=SseConnectionParams(
        url="http://localhost:3000/sse",
        timeout=10.0,
        sse_read_timeout=300.0,
    )
)
```

### mcp_progress_callback_agent

```python
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp.shared.session import ProgressFnT

async def simple_progress_callback(progress: float, total: float | None, message: str | None) -> None:
  if total is not None:
    percentage = (progress / total) * 100
    print(f"[...] {percentage:.0f}% ({progress}/{total}) {message or ''}")
  else:
    print(f"Progress: {progress} {f'- {message}' if message else ''}")


def progress_callback_factory(
    tool_name: str,
    *,
    callback_context: CallbackContext | None = None,
    **kwargs: Any,
) -> ProgressFnT | None:
  session_id = callback_context.session.id if callback_context else "unknown"

  async def callback(progress: float, total: float | None, message: str | None) -> None:
    if callback_context:
      callback_context.state["last_progress"] = progress
      callback_context.state["last_total"] = total
    # print progress...

  return callback


root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="progress_demo_agent",
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(...),
            progress_callback=progress_callback_factory,  # or simple_progress_callback
        )
    ],
)
```

**MCP Progress Callback APIs**:
- `McpToolset(connection_params=..., progress_callback=callback_or_factory)`
- Simple callback: `async def callback(progress: float, total: float | None, message: str | None) -> None`
- Factory (for per-tool callbacks with context): `def factory(tool_name: str, *, callback_context: CallbackContext | None = None, **kwargs) -> ProgressFnT | None`
- Factory receives `CallbackContext` — can read/write session state
- `**kwargs` required for forward compatibility

---

## 25. json_passing_agent (Sequential with output_key + Pydantic)

```python
from google.adk import Agent
from google.adk.agents import sequential_agent
from google.adk.tools import tool_context
from pydantic import BaseModel

SequentialAgent = sequential_agent.SequentialAgent
ToolContext = tool_context.ToolContext


class PizzaOrder(BaseModel):
  size: str
  crust: str
  toppings: list[str]


order_intake_agent = Agent(
    name='order_intake_agent',
    model='gemini-2.5-flash',
    instruction='...',
    output_key='pizza_order',
    output_schema=PizzaOrder,
    tools=[get_available_sizes, get_available_crusts, get_available_toppings],
)


def calculate_price(tool_context: ToolContext) -> str:
  order_dict = tool_context.state.get('pizza_order')
  if not order_dict:
    return "I can't find an order to calculate the price for."
  order = PizzaOrder.model_validate(order_dict)
  # ... compute price ...


order_confirmation_agent = Agent(
    name='order_confirmation_agent',
    model='gemini-2.5-flash',
    instruction='The order is in the state variable `pizza_order`. First, use `calculate_price` tool...',
    tools=[calculate_price],
)

root_agent = SequentialAgent(
    name='pizza_ordering_agent',
    sub_agents=[order_intake_agent, order_confirmation_agent],
)
```

**Pattern**: JSON/Pydantic passing between sequential agents via state:
1. First agent has `output_key='pizza_order'` and `output_schema=PizzaOrder` → stores structured data in state
2. Second agent's tools read `tool_context.state.get('pizza_order')` → dict
3. Use `PizzaOrder.model_validate(order_dict)` to convert back to typed Pydantic model

---

## 26. pydantic_argument (FunctionTool with Union/Optional Pydantic args)

```python
from typing import Optional, Union
from google.adk.agents.llm_agent import Agent
from google.adk.tools.function_tool import FunctionTool
import pydantic


class UserProfile(pydantic.BaseModel):
  name: str
  age: int
  email: Optional[str] = None


class UserPreferences(pydantic.BaseModel):
  theme: str = "light"
  language: str = "English"
  notifications_enabled: bool = True


class CompanyProfile(pydantic.BaseModel):
  company_name: str
  industry: str
  employee_count: int
  website: Optional[str] = None


def create_full_user_account(
    profile: UserProfile, preferences: Optional[UserPreferences] = None
) -> dict:
  # FunctionTool auto-converts JSON dicts to Pydantic instances
  ...


def create_entity_profile(entity: Union[UserProfile, CompanyProfile]) -> dict:
  # FunctionTool handles Union types automatically
  if isinstance(entity, UserProfile): ...
  elif isinstance(entity, CompanyProfile): ...


root_agent = Agent(
    model="gemini-2.5-pro",
    name="profile_agent",
    ...
    tools=[
        FunctionTool(create_full_user_account),
        FunctionTool(create_entity_profile),
    ],
)
```

**Pattern**: `FunctionTool` automatically handles conversion of JSON dicts (from LLM) to Pydantic models, including `Optional[Model]` and `Union[Model1, Model2]` types. No manual conversion needed.

---

## 27. static_instruction (Static + Dynamic instruction)

```python
from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types


STATIC_INSTRUCTION_TEXT = """You are Bingo, a lovable digital pet companion!..."""


def provide_dynamic_instruction(ctx: ReadonlyContext | None = None):
  """Provides dynamic hunger-based instructions."""
  hunger_level = "starving"
  if ctx:
    session = ctx._invocation_context.session
    if session and session.state:
      last_fed = session.state.get("last_fed_timestamp")
      if last_fed:
        hunger_level = get_hunger_state(last_fed)
  return f"""CURRENT HUNGER STATE: {hunger_level}\n..."""


root_agent = Agent(
    model="gemini-2.5-flash",
    name="bingo_digital_pet",
    static_instruction=types.Content(
        role="user", parts=[types.Part(text=STATIC_INSTRUCTION_TEXT)]
    ),
    instruction=provide_dynamic_instruction,  # callable: called every turn
    tools=[eat],
)
```

**Pattern**: Two-tier instruction:
- `static_instruction` — passed as `types.Content`, cached for performance (context caching)
- `instruction` — can be a string OR a callable `(ctx: ReadonlyContext | None) -> str` called every turn
- `ReadonlyContext` provides read access to session state without allowing mutation

---

## 28. live_bidi_streaming_multi_agent (Live API / Gemini Native Audio)

```python
from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types


roll_agent = Agent(
    name="roll_agent",
    model=Gemini(
        model="gemini-live-2.5-flash-native-audio",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore",
                )
            )
        ),
    ),
    ...
)

root_agent = Agent(
    model=Gemini(
        model="gemini-live-2.5-flash-native-audio",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Zephyr",
                )
            )
        ),
    ),
    sub_agents=[roll_agent, prime_agent],
    tools=[get_current_weather],
    ...
)
```

**Pattern**: Multi-agent with Gemini Live (real-time bidirectional audio). Each agent gets its own voice.

**Key APIs**:
- `from google.adk.models.google_llm import Gemini`
- `Gemini(model="gemini-live-2.5-flash-native-audio", speech_config=types.SpeechConfig(...))`
- `types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")))`

---

## 29. interactions_api (Gemini Interactions API)

```python
from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.google_search_tool import GoogleSearchTool


root_agent = Agent(
    model=Gemini(
        model="gemini-2.5-flash",
        use_interactions_api=True,
    ),
    name="interactions_test_agent",
    tools=[
        GoogleSearchTool(bypass_multi_tools_limit=True),
        get_current_weather,
    ],
)
```

**Pattern**: Use Gemini Interactions API by setting `use_interactions_api=True` on the `Gemini` model config.
- `GoogleSearchTool(bypass_multi_tools_limit=True)` — converts built-in search to function calling, enabling mixing with custom tools

---

## 30. custom_code_execution (Custom Code Executor)

```python
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.code_executors.code_execution_utils import CodeExecutionInput, CodeExecutionResult, File
from google.adk.code_executors.vertex_ai_code_executor import VertexAiCodeExecutor
from typing_extensions import override


class CustomCodeExecutor(VertexAiCodeExecutor):
  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    font_file = _load_font_file(font_url, font_filename)
    if font_file is not None:
      code_execution_input.input_files.append(font_file)
      code_execution_input.code = f"{_FONT_SETUP_CODE}\n\n{code_execution_input.code}"
    return super().execute_code(invocation_context, code_execution_input)


root_agent = Agent(
    model="gemini-2.5-flash",
    name="data_science_agent",
    instruction=base_system_instruction() + "...",
    code_executor=CustomCodeExecutor(),
)
```

**Pattern**: Custom code executor by subclassing `VertexAiCodeExecutor` and overriding `execute_code`. Inject files or modify code before execution. Register via `Agent(code_executor=CustomCodeExecutor())`.

---

## 31. artifact_save_text

```python
from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types


async def log_query(tool_context: ToolContext, query: str):
  """Saves the provided query string as a 'text/plain' artifact named 'query'."""
  query_bytes = query.encode('utf-8')
  artifact_part = types.Part(
      inline_data=types.Blob(mime_type='text/plain', data=query_bytes)
  )
  await tool_context.save_artifact('query', artifact_part)


root_agent = Agent(
    model='gemini-2.0-flash',
    name='log_agent',
    ...
    tools=[log_query],
)
```

---

## 32. postgres_session_service

```python
from google.adk.agents.llm_agent import Agent


root_agent = Agent(
    model="gemini-2.0-flash",
    name="postgres_session_agent",
    description="A sample agent demonstrating PostgreSQL session persistence.",
    instruction="""
      You are a helpful assistant that demonstrates session persistence.
      You can remember previous conversations within the same session.
      ...
    """,
    tools=[get_current_time],
)
```

**Pattern**: Agent itself is simple; PostgreSQL session persistence is configured at the runner/app level (not in agent.py).

---

## Key API Reference Summary

### Core Classes

```python
# Agent / LlmAgent
from google.adk import Agent
from google.adk.agents.llm_agent import LlmAgent  # same thing

# Workflow Agents
from google.adk.agents.sequential_agent import SequentialAgent
# LoopAgent not found in samples — likely: from google.adk.agents.loop_agent import LoopAgent

# Remote A2A
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH

# App & Resumability
from google.adk.apps import App, ResumabilityConfig

# Runners
from google.adk.runners import InMemoryRunner
```

### Tools

```python
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.example_tool import ExampleTool
from google.adk.tools.load_memory_tool import load_memory_tool
from google.adk.tools.preload_memory_tool import preload_memory_tool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams, StdioConnectionParams
from google.adk import AgentTool  # wraps sub-agent as a tool
from google.adk.tools.langchain_tool import LangchainTool
```

### Models

```python
from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
from google.adk.models.anthropic_llm import Claude
from google.adk.models.google_llm import Gemini
```

### Callbacks

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
```

### Planners

```python
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
```

### Plugins

```python
from google.adk.plugins.base_plugin import BasePlugin
```

### Code Executors

```python
from google.adk.code_executors.vertex_ai_code_executor import VertexAiCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput, CodeExecutionResult, File
```

---

## Callback Signature Quick Reference

```python
# Agent lifecycle
async def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    ...  # return ModelContent to short-circuit

async def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    ...  # return ModelContent to replace output

# Model lifecycle
async def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    ...  # return LlmResponse to skip model call
    # mutate llm_request.contents to modify history

async def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    ...  # return new LlmResponse to replace

# Tool lifecycle
def before_tool_callback(tool, args: dict, tool_context: ToolContext) -> Optional[dict]:
    ...  # return dict to skip tool and use as response

def after_tool_callback(tool, args: dict, tool_context: ToolContext, tool_response: dict) -> Optional[dict]:
    ...  # return dict to override tool response

# Plugin lifecycle (keyword-only args)
async def before_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> None
async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> None
```

---

## State Access Patterns

```python
# In tool functions
def my_tool(tool_context: ToolContext) -> dict:
    tool_context.state['key'] = value          # write
    val = tool_context.state.get('key')        # read
    val = tool_context.state['key']            # read (raises if missing)

    # Artifact operations
    version = await tool_context.save_artifact('filename', types.Part(...))
    artifact = await tool_context.load_artifact('filename')  # returns Part or None

    # Human-in-loop
    confirmation = tool_context.tool_confirmation    # ToolConfirmation or None
    tool_context.request_confirmation(hint='...', payload={'key': 'value'})

# In callbacks
def my_callback(callback_context: CallbackContext, ...):
    callback_context.state['key'] = value
    val = callback_context.state.get('key')
    session = callback_context._invocation_context.session
    session_service = callback_context._invocation_context.session_service

# IMPORTANT: State mutation pattern (avoid in-place mutation of lists/dicts)
tool_context.state['list'] = tool_context.state['list'] + [new_item]  # CORRECT
tool_context.state['list'].append(new_item)  # may not persist correctly
```

---

## LiteLlm Configuration Summary

```python
from google.adk.models.lite_llm import LiteLlm, LiteLLMClient

# Basic usage — any LiteLLM provider
LiteLlm(model="openai/gpt-4o")
LiteLlm(model="anthropic/claude-3-sonnet-20240229")
LiteLlm(model="gemini/gemini-2.5-flash")
LiteLlm(model="vertex_ai/gemini-2.5-pro-exp-03-25")
LiteLlm(model="vertex_ai/claude-3-5-haiku")

# With fallbacks
LiteLlm(
    model='gemini/gemini-2.5-pro',
    fallbacks=['anthropic/claude-sonnet-4-5-20250929', 'openai/gpt-4o'],
)

# Inject tool schemas manually (for models that need it)
LiteLlm(
    model="vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas",
    functions=[convert_to_openai_function(my_func), ...],
)

# Custom backend
LiteLlm(model="mock/my-model", llm_client=MyCustomLiteLLMClient())

# Custom client — subclass LiteLLMClient
class MyClient(LiteLLMClient):
    async def acompletion(self, model, messages, tools, **kwargs) -> dict:
        # Return OpenAI-format completion response
        return {
            "id": "...",
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
```

---

## Human Tool Confirmation — Complete Flow

```python
# Pattern A: FunctionTool with require_confirmation
FunctionTool(my_func, require_confirmation=True)                    # always confirm
FunctionTool(my_func, require_confirmation=async_callable)         # conditional confirm

# async callable signature:
async def should_confirm(arg1, arg2, ..., tool_context: ToolContext) -> bool:
    return True  # or False

# Pattern B: Manual request_confirmation inside tool
def my_tool(tool_context: ToolContext, param: str):
    confirmation = tool_context.tool_confirmation
    if confirmation is None:
        # First call — request confirmation
        tool_context.request_confirmation(
            hint='Human-readable instruction for the approver',
            payload={'default_value': 'something'},
        )
        return {'status': 'pending, waiting for confirmation'}

    # Resumed call — use confirmation
    approved_value = confirmation.payload['key']
    return {'status': 'done', 'value': approved_value}

# App must have resumability enabled
app = App(
    name='my_app',
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```
