# grasp_tree.py

from copy import deepcopy
import random
from typing import List, Set, Dict, Optional

from .utils import ObjectTag, ObjectTagWithCount, ActionFailed, item_id_to_str
from .room_tree import RoomTree


class GraspTree:
    def __init__(self, room_tree: RoomTree):
        self.room_tree = room_tree
        self.item_accesses = {}  # item_id -> [AccessPath]
        self.item_difficulty = {}  # item_id -> difficulty_score
        self._build_item_access_paths()

    def _build_item_access_paths(self):
        """
        构建每个物品可以被访问的所有路径（包括房间、家具等）
        """
        for room_id in self.room_tree.rooms:
            room = self.room_tree.rooms[room_id]
            for furniture in room.furnitures:
                for obj in furniture.objects:
                    item_id = obj.object_tag.object_id
                    path = {
                        "room": room_id,
                        "furniture": furniture.furniture_id,
                        "open_required": furniture.openable,
                    }
                    if item_id not in self.item_accesses:
                        self.item_accesses[item_id] = []
                    self.item_accesses[item_id].append(path)

        # 计算每个物品的抓取难度
        for item_id in self.item_accesses:
            self.item_difficulty[item_id] = self._calculate_item_difficulty(item_id)

    def _calculate_item_difficulty(self, item_id: str) -> int:
        """
        根据访问路径计算抓取难度
        """
        paths = self.item_accesses.get(item_id, [])
        if not paths:
            return float('inf')  # 不可抓取

        min_difficulty = float('inf')
        for path in paths:
            difficulty = 0
            if path["furniture"]:
                difficulty += 1  # 需要移动到家具
                if path["open_required"]:
                    difficulty += 1  # 需要打开家具
            if self.room_tree.rooms[path["room"]].door_open:
                difficulty += 0
            else:
                difficulty += 1  # 需要开门
            min_difficulty = min(min_difficulty, difficulty)
        return min_difficulty

    def get_grasp_difficulty(self, item_id: str) -> int:
        """
        获取某个物品的抓取难度
        """
        return self.item_difficulty.get(item_id, float('inf'))

    def items_with_min_difficulty(self, min_difficulty=1):
        """
        返回满足最低难度的物品列表
        """
        return [
            (item_id, difficulty)
            for item_id, difficulty in self.item_difficulty.items()
            if difficulty >= min_difficulty
        ]

    def create_task_set(self, goal_item: str):
        """
        根据目标物品生成任务描述和可用动作
        """
        task_description = f"Goal: 找到并抓取 {item_id_to_str(goal_item)}。\n\n"
        possible_rooms = self.room_tree.create_room_set(goal_item)
        task_description += f"可用房间：{', '.join(possible_rooms)}\n"

        actions = []
        distractors = []

        for room in possible_rooms:
            actions.append(f"move to {room}")
            actions.append(f"open {room}")
            actions.append(f"enter {room}")
            for fid in self.room_tree.rooms[room].furnitures_dict.keys():
                actions.append(f"move to {fid}")
                actions.append(f"open {fid}")
                actions.append(f"grasp {goal_item}")

        # 构造干扰动作
        all_items = list(self.room_tree.itemid_set)
        distractors = [f"grasp {item}" for item in all_items if item != goal_item]

        sampled_actions = random.sample(actions + distractors, min(len(actions + distractors), 15))
        task_description += "可能的动作：" + ", ".join(sampled_actions)

        return task_description

    def is_valid_grasp_action(self, action: str) -> bool:
        """
        判断一个抓取动作是否合法
        """
        match = re.match(r"grasp (.*)", action.lower())
        if not match:
            return False
        item_name = match.group(1).strip()
        return item_name in self.room_tree.itemid_set

    def traverse_grasp_tree(self, item_id: str, visited_items: Optional[Set[str]] = None):
        """
        类似 crafting_tree 的 traverse_recipe_tree
        """
        if visited_items is None:
            visited_items = set()

        if item_id in visited_items:
            raise ValueError(f"循环依赖 detected for item {item_id}: {visited_items}")

        visited_items.add(item_id)
        paths = self.item_accesses.get(item_id, [])

        subtasks = []
        for path in paths:
            room_id = path["room"]
            furniture_id = path["furniture"]

            if furniture_id:
                subtasks.append(f"open {furniture_id}")
            subtasks.append(f"grasp {item_id}")

        return subtasks

    def collect_item_accesses(self, item_id: str):
        """
        收集该物品的所有抓取路径
        """
        return self.item_accesses.get(item_id, [])

    def sample_easy_goal(self):
        """
        随机采样一个简单的目标物品
        """
        easy_items = [item for item, d in self.item_difficulty.items() if d == 1]
        return random.choice(easy_items) if easy_items else None

    def sample_hard_goal(self):
        """
        随机采样一个困难的目标物品
        """
        hard_items = sorted(self.item_difficulty.items(), key=lambda x: x[1], reverse=True)
        return hard_items[0][0] if hard_items else None