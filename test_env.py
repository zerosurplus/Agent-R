from agentenv.envs import RoomGraspEnvClient
from fastchat.model.model_adapter import get_conversation_template
import os
from mcts_utils.llm_server import FuncCallOffline

# 初始化环境客户端
env = RoomGraspEnvClient(env_server_base="http://localhost:36001", data_len=200)

# 输出环境ID和初始观察
print("Environment ID:", env.env_id)
obs = env.reset(25)
print("reset 返回值:", obs)
print("env.info['observation']:", env.info["observation"])
print("Initial Observation:", obs)


def setup_conversation(env):
    """
    用于测试流程中构造对话历史的辅助函数
    """
    print(f"[DEBUG] env.conversation_start: {env.conversation_start}")

    conversation = list(env.conversation_start)
    conv = get_conversation_template('gpt-4')

    conv.append_message(conv.roles[0], conversation[0]["value"])
    conv.append_message(conv.roles[1], 'Ok.')

    task_type = os.environ.get("TASK", "unknown")
    print(f"[DEBUG] Current Task Type: {task_type}")

    if task_type == "webshop":
        initial_obs = env.observe()
    else:
        initial_obs = env.info["observation"]

    conv.append_message(conv.roles[0], initial_obs)

    print(f"[DEBUG] Final conv.messages: {conv.messages}")
    print(f"[DEBUG] Initial Observation: {initial_obs}")

    return conv


conv = setup_conversation(env)
print("Conversation Messages:", conv.messages)

calling = FuncCallOffline(model_name="Llama-3.1-8B-Instruct")

max_steps = 100
done = False
step = 0

while not done and step < max_steps:
    # 构造 prompt
    prompt = [{"role": role, "content": content} for role, content in conv.messages]

    # LLM 推理
    response = calling.llm_func(prompt, model_name="Llama-3.1-8B-Instruct")

    new_action = response.split("Action:")[-1].strip()

    print(f"\nExtracted Action: {new_action}")

    # 环境 step
    step_output = env.step(new_action)
    obs = step_output.state
    reward = step_output.reward
    done = step_output.done

    print(f"\nStep {step}:")
    print(f"Action       : {new_action}")
    print(f"Observation  : {obs}")
    print(f"Done         : {done}\n")

    # 更新对话历史
    conv.append_message(conv.roles[1], None)
    conv.update_last_message(response)
    conv.append_message(conv.roles[0], obs)

    step += 1