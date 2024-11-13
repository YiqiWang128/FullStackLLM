from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.core.base.llms.types import MessageRole, ChatMessage
import os
import warnings

# 设置 pydantic 警告过滤
warnings.filterwarnings("ignore", message=r'.*Field "model_name" in DashScope.*')

# 初始化 DashScope LLM
try:
    dashscope_llm = DashScope(
        model_name=DashScopeGenerationModels.QWEN_PLUS,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    # 初始化消息列表，支持多轮对话
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant.")
    ]

    def chat_with_user(user_input, messages, stream=False):
        # 将用户输入添加到消息列表
        messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
        
        if stream:
            # 流式输出模式
            responses = dashscope_llm.stream_chat(messages)
            try:
                for response in responses:
                    if hasattr(response, 'delta'):
                        print(response.delta, end="", flush=True)  # 确保每次增量输出后立即显示
            except Exception as stream_error:
                print(f"流式输出错误: {stream_error}")
            print()  # 每轮对话结束后再换行
        else:
            # 非流式输出模式
            resp = dashscope_llm.chat(messages)
            print(resp.response.message.content)
            print()  # 每轮对话结束后换行

        # 检查用户是否希望结束对话
        if user_input.lower() in ["结束对话", "再见", "退出"]:
            print("感谢您的咨询，再见")
            return False
        return True

    # 示例对话，用户可以多次输入内容
    while True:
        user_input = input("用户: ")
        if not chat_with_user(user_input, messages, stream=True):  # 设置 stream=True 以使用流式输出
            break

except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
