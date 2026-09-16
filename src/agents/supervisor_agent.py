from langchain.agents import create_agent
from langchain_community.chat_models import ChatOpenAI
from langgraph.checkpoint.redis import AsyncRedisSaver

from core.config import get_settings
from src.infra.redis_cache import get_checkpointer_redis

async def create_supervisor_agent():
    settings = get_settings()
    redis_client = get_checkpointer_redis()

    checkpointer = AsyncRedisSaver(redis_client=redis_client)
    await checkpointer.asetup()

    llm = ChatOpenAI(
        base_url=settings.BASE_URL_CHAT,
        api_key=settings.API_KEY_CHAT,
        model=settings.CHAT_MODEL
    )

    agent = create_agent(
        model=llm,
        tools=[],
        checkpointer=checkpointer # 短期记忆
    )

    return agent


_supervisor_agent = None
async def get_supervisor_agent():
    """返回全局单例 Agent，首次调用初始化"""
    global _supervisor_agent

    if _supervisor_agent is None:
        _supervisor_agent = await create_supervisor_agent()

    return _supervisor_agent


async def chat_endpoint(user_id: str, session_id: str, message: str):
    agent = await get_supervisor_agent()

    config = {
        "configurable": {
            "thread_id": f"{user_id}:{session_id}"
        }
    }

    result = await agent.ainvoke(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        },
        config=config
    )

    return result['messages'][-1].content