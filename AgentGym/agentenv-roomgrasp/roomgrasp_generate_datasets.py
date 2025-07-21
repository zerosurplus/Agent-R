# roomgrasp_generate_datasets.py
"""
自动生成 roomgrasp 的训练集和测试集 JSON 文件。
"""

import json
from agentenv_roomgrasp.room_tree import RoomTree
from agentenv_roomgrasp.grasp_tree import GraspTree
import os

def generate_train_clean_json(output_path="/remote-home/2432192/Agent-R/mcts_utils/roomgrasp/roomgrasp_train_clean.json"):
    """
    生成 roomgrasp_train_clean.json，格式为 {index: difficulty}
    """
    # 初始化房间树和抓取树
    rt = RoomTree(room_config_dir="agentenv_roomgrasp/")
    gt = GraspTree(rt)

    task_list = []

    # 收集所有可抓取物品及其难度
    item_difficulties = {}
    for room in rt.rooms:
        room_obj = rt.rooms[room]
        for furniture in room_obj.furnitures:
            for obj in furniture.objects:
                item_id = obj.object_tag.object_id
                if item_id not in item_difficulties:
                    difficulty = gt.get_grasp_difficulty(item_id)
                    item_difficulties[item_id] = difficulty
                    task_list.append({"room": room, "object": item_id})
                    print(f"Task {len(task_list)-1}: {room} - {item_id}, Difficulty: {difficulty}")

    # 写入 train_clean.json
    with open(output_path, "w") as f:
        json.dump({str(i): item_difficulties[task["object"]] for i, task in enumerate(task_list)}, f, indent=2)

    print(f"✅ Generated {output_path}")
    return len(task_list)


def generate_test_json(total_tasks, output_path="/remote-home/2432192/Agent-R/mcts_utils/roomgrasp/roomgrasp_test.json"):
    """
    生成 roomgrasp_test.json，格式为 [{"item_id": "roomgrasp_0"}, ...]
    """
    test_tasks = [{"item_id": f"roomgrasp_{i}"} for i in range(total_tasks)]

    with open(output_path, "w") as f:
        json.dump(test_tasks, f, indent=2)

    print(f"✅ Generated {output_path}")


if __name__ == "__main__":
    os.makedirs("mcts_utils/roomgrasp", exist_ok=True)

    # Step 1: 生成训练集（带难度值）
    total_tasks = generate_train_clean_json()

    # Step 2: 生成测试集（对应任务编号）
    generate_test_json(total_tasks)