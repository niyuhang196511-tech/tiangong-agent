import pytest

from agents.supervisor_agent import chat_endpoint

@pytest.mark.asyncio
async def test_supervisor_agent():
    result = await chat_endpoint("123", "24224", "你好，我是大帅哥。")
    print(result)
    result = await chat_endpoint("123", "24224", "我是谁？")
    print(result)