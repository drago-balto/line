"""
Transfer After Speech Example - Demonstrates waiting for TTS to finish before transferring.

The agent is instructed to say a long farewell phrase while calling transfer_call in the same turn.
Set WAIT_FOR_SPEECH=true (default) to wait for speech to complete before transferring.
Set WAIT_FOR_SPEECH=false to transfer immediately (cutting off the agent mid-sentence).

Run with: ANTHROPIC_API_KEY=your-key TRANSFER_TO=+14155551234 uv run python main.py
Compare: ANTHROPIC_API_KEY=your-key TRANSFER_TO=+14155551234 WAIT_FOR_SPEECH=false uv run python main.py
"""

import os
from typing import Annotated

import phonenumbers

from line import AgentSendText, AgentTransferCall
from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, passthrough_tool
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

WAIT_FOR_SPEECH = os.getenv("WAIT_FOR_SPEECH", "true").lower() != "false"
TRANSFER_TO = os.getenv("TRANSFER_TO", "")

SYSTEM_PROMPT = """\
You are a friendly assistant that helps callers with quick questions.

When the user asks to be transferred or says they want to speak to someone else, you MUST:
1. Say a long, warm message before transferring. Be elaborate - thank them for calling, let them \
know you're transferring them now, wish them a wonderful rest of their day, and reassure them \
that the person they're being connected to will take great care of them. Use at least 3-4 sentences.
2. Call the transfer_call tool in the SAME response as your transfer message.

This is important: your speech and the transfer_call tool call must happen together in the \
same turn. Do not transfer in a separate turn from your message."""

INTRODUCTION = (
    "Hi there! I'm here to help with any quick questions. "
    "Just ask to be transferred whenever you'd like to speak to someone else."
)


@passthrough_tool
async def transfer_call(
    ctx: ToolEnv,
    reason: Annotated[str, "The reason for transferring the call"],
):
    """Transfer the call to another person. Use when the user asks to be transferred."""
    parsed = phonenumbers.parse(TRANSFER_TO)
    normalized = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    yield AgentSendText(text=f"Transferring you now to {normalized}.")
    yield AgentTransferCall(target_phone_number=normalized, after_speech=WAIT_FOR_SPEECH)


async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[transfer_call],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    if not TRANSFER_TO:
        print("ERROR: Set TRANSFER_TO env var to a phone number (e.g. +14155551234)")
        exit(1)
    mode = "waiting for speech" if WAIT_FOR_SPEECH else "immediate (no wait)"
    print(f"Starting Transfer After Speech example (mode: {mode}, transfer to: {TRANSFER_TO})")
    app.run()
