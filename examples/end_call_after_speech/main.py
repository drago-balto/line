"""
End Call After Speech Example - Demonstrates waiting for TTS to finish before ending the call.

The agent is instructed to say a long farewell phrase while calling end_call in the same turn.
Set WAIT_FOR_SPEECH=true (default) to wait for speech to complete before ending.
Set WAIT_FOR_SPEECH=false to end immediately (cutting off the agent mid-sentence).

Run with: ANTHROPIC_API_KEY=your-key uv run python main.py
Compare: ANTHROPIC_API_KEY=your-key WAIT_FOR_SPEECH=false uv run python main.py
"""

import os
from typing import Annotated

from line import AgentEndCall
from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, passthrough_tool
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

WAIT_FOR_SPEECH = os.getenv("WAIT_FOR_SPEECH", "true").lower() != "false"

SYSTEM_PROMPT = """\
You are a friendly assistant that helps callers with quick questions.

When the user says goodbye or wants to end the call, you MUST:
1. Say a long, warm farewell. Be elaborate - thank them for calling, wish them a wonderful rest \
of their day, remind them they can call back anytime, and say a heartfelt goodbye. Use at least \
3-4 sentences.
2. Call the end_call tool in the SAME response as your farewell message.

This is important: your farewell speech and the end_call tool call must happen together in the \
same turn. Do not end the call in a separate turn from your farewell."""

INTRODUCTION = (
    "Hi there! I'm here to help with any quick questions. Just say goodbye whenever you're ready to hang up."
)


@passthrough_tool
async def end_call(
    ctx: ToolEnv,
    reason: Annotated[str, "The reason for ending the call"],
):
    """End the call. Use when the user says goodbye or signals they're finished."""
    yield AgentEndCall(after_speech=WAIT_FOR_SPEECH)


async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[end_call],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    mode = "waiting for speech" if WAIT_FOR_SPEECH else "immediate (no wait)"
    print(f"Starting End Call After Speech example (mode: {mode})")
    app.run()
