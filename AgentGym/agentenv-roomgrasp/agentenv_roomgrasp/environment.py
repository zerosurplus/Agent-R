import random
import gymnasium as gym
import re
from .utils import ActionFailed, item_id_to_str, ObjectTag, ObjectTagWithCount, RoomStructure
from .room_tree import RoomTree
from typing import List
import logging
import re

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoomGraspEnv(gym.Env[str, str]):
    """
    房间探索与抓取任务环境
    动作空间：
        Move <room> - 移动到某个房间门前
        Open <room> - 打开指定房门
        Enter <room> - 进入已开门的房间
        Open <furniture> - 打开家具
        Grasp <object> - 抓取当前房间或家具内的物品
        Inventory - 查看背包
    观测空间：
        文本描述 + 背包信息
    """

    def __init__(self, room_tree: RoomTree, commands: str, goal: str):
        self.inventory = {}
        self.action_regexes = {
            "move": r"move\s+(\w+)",
            "open": r"open\s+(\w+)",     # 统一 open
            "enter": r"enter\s+(\w+)",
            "grasp": r"grasp (.*)",
            "inventory": r"inventory",
            "exit": r"exit",
        }
        logger.info(f"[DEBUG] Initializing env with goal: {goal}")
        self.count_regex = r"([0-9]+) (.*)"
        self.room_tree = room_tree
        self.commands = commands
        self.goal = goal
        self.state = "out_of_room"  # 可选值：out_of_room, in_room, near_furniture
        self.current_room = None     # 当前所在房间
        self.current_location = None # 当前所在家具
        self.room_states = {
            room_id: room_tree._create_room_state(room_structure)
            for room_id, room_structure in room_tree.rooms.items()
        }

    def step(self, action):
        observation = None
        reward = 0
        terminated = False
        truncated = False
        info = {}

        try:
            for action_type, regex in self.action_regexes.items():
                match = re.match(regex, action.lower())
                if match:
                    logger.info(f"[DEBUG] Matched action type: {action_type}")  # 新增
                    if action_type == "move":
                        target = match.group(1).strip()
                        if self.room_tree.is_valid_room(target):
                            if self.state == "in_room":
                                raise ActionFailed("You must exit current room before moving to another room")
                            self.current_room = None
                            self.current_location = target
                            self.state = "out_of_room"
                            door_open = self.room_states[target].door_open
                            observation = f"You arrived at the door of {target}, which is {'opened' if door_open else 'closed'}."
                        elif self.state in ["in_room", "near_furniture"]:
                            room_state = self.room_states[self.current_room]
                            room_obj = self.room_tree.rooms[self.current_room]
                            furnitures_in_room = list(room_state.furnitures.keys())
                            
                            all_furniture_names = set()
                            for r_id, r_state in self.room_states.items():
                                all_furniture_names.update(r_state.furnitures.keys())

                            if target in furnitures_in_room:
                                self.current_location = target
                                self.state = "near_furniture"

                                furniture_state = room_state.furnitures[target]
                                furniture_data = room_obj.furnitures_dict[target]

                                if not furniture_data.openable:
                                    items = ", ".join([obj.object_tag.object_id for obj in furniture_state.objects])
                                    observation = f"You arrived next to {target}. It contains: {items if items else 'nothing'}."
                                elif furniture_state.opened:
                                    items = ", ".join([obj.object_tag.object_id for obj in furniture_state.objects])
                                    observation = f"You arrived next to {target}, which is opened. It contains: {items if items else 'nothing'}."
                                else:
                                    observation = f"You arrived next to {target}, which is closed."

                            elif target in all_furniture_names:
                                raise ActionFailed(f"{target} belongs to another room. Please enter the corresponding room first.")
                            else:
                                raise ActionFailed(f"{target} is not a valid furniture.")
                        else:
                            raise ActionFailed(f"Cannot move to {target}")

                    elif action_type == "open":
                        target = match.group(1).strip()

                        # 判断是否是房间
                        if self.room_tree.is_valid_room(target):
                            if self.state != "out_of_room" or self.current_location != target:
                                raise ActionFailed(f"You are not at the door of {target}, cannot open it")

                            room_state = self.room_states[target]
                            if room_state.door_open:
                                raise ActionFailed(f"The door of {target} is already opened")
                            room_state.door_open = True
                            observation = f"You opened the door of {target}"

                        # 否则视为家具
                        elif self.state == "near_furniture" and self.current_location == target:
                            room_state = self.room_states[self.current_room]
                            furniture_state = room_state.furnitures.get(target, None)
                            room_obj = self.room_tree.rooms[self.current_room]  
                            furniture = room_obj.furnitures_dict.get(target, None)
                            if not furniture_state:
                                raise ActionFailed(f"{target} not a valid furniture")
                            if not furniture.openable:
                                raise ActionFailed(f"{target} cannot be opened")
                            if furniture.opened:
                                raise ActionFailed(f"{target} is already opened")

                            furniture_state.opened = True
                            observation = f"You opened {target}"
                            item_list = ", ".join([obj.object_tag.object_id for obj in furniture_state.objects])
                            observation += f". It contains: {item_list}"

                        else:
                            raise ActionFailed(f"Cannot open {target}, you are not near it")

                    elif action_type == "enter":
                        target = match.group(1).strip()
                        if not self.room_tree.is_valid_room(target):
                            raise ActionFailed(f"{target} not a valid room")
                        if self.state != "out_of_room" or self.current_location != target:
                            raise ActionFailed(f"You are not at the door of {target}, cannot enter")

                        room_state = self.room_states[target]
                        if not room_state.door_open:
                            raise ActionFailed(f"The door of {target} is closed, please open it first")

                        self.current_room = target
                        self.current_location = None
                        self.state = "in_room"
                        observation = f"You entered {target}"
                        # 调试输出
                        furnitures_in_room = list(self.room_states[self.current_room].furnitures.keys())
                        observation = f"You entered {target} (accessible furniture: {', '.join(furnitures_in_room)})"

                    elif action_type == "exit":
                        if self.state not in ["in_room", "near_furniture"]:
                            raise ActionFailed("You are not in a room, cannot exit")

                        self.current_room = None
                        self.current_location = None
                        self.state = "out_of_room"
                        observation = "You exited the room"

                    elif action_type == "grasp":
                        logger.info("[DEBUG] Entering grasp action handler")
                        if self.state != "near_furniture":
                            raise ActionFailed("Please move to a furniture before attempting to grasp an item.")

                        furniture_name = self.current_location
                        room_state = self.room_states[self.current_room]
                        furniture_state = room_state.furnitures.get(furniture_name, None)

                        if not furniture_state:
                            raise ActionFailed(f"{furniture_name} is not a valid furniture.")

                        furniture_data = self.room_tree.rooms[self.current_room].furnitures_dict[furniture_name]

                        if furniture_data.openable and not furniture_state.opened:
                            raise ActionFailed(f"{furniture_name} is closed. Please open it first.")

                        target = match.group(1).strip()

                        found = False
                        for obj in furniture_state.objects:
                            if obj.object_tag.object_id == target or self.room_tree.is_item_in_tag(target, obj.object_tag.object_id):
                                furniture_state.objects.remove(obj)
                                found = True
                                break

                        if not found:
                            raise ActionFailed(f"{furniture_name} does not contain {target}.")

                        self.add_item(target, 1)
                        observation = f"You grasped {item_id_to_str(target)}"

                        if target == self.goal or item_id_to_str(target) == self.goal or self.room_tree.is_item_in_tag(self.goal, target):
                            reward = 1.0
                            terminated = True

                    elif action_type == "inventory":
                        item_names = [
                            item_id_to_str(item_id) for item_id in self.inventory.keys()
                        ]
                        items_str = ", ".join(item_names) if item_names else "empty"
                        observation = f"Your inventory contains: {items_str}"

                    else:
                        raise NotImplementedError(f"Action type {action_type} is not implemented.")

            if observation is None:
                raise ActionFailed(f"Failed to execute action: {action}")

        except ActionFailed as e:
            observation = str(e)
            logger.warning(f"[Action Failed] {e}")
        except Exception as e:
            observation = f"System Error: {str(e)}"
            logger.error(f"[Environment Error] {e}", exc_info=True)

        return (observation, reward, terminated, truncated, info)

    def add_item(self, item_id: str, amt: int):
        if item_id not in self.inventory:
            self.inventory[item_id] = 0
        self.inventory[item_id] += amt

    def reset(self, seed=42, data_idx=0, commands=None, goal=None):
        super().reset(seed=seed)
        self.inventory.clear()
        self.state = "out_of_room"
        self.current_room = None
        self.current_location = None
        # 重置所有房间的状态为初始状态
        self.room_states = {
            room_id: self.room_tree._create_room_state(self.room_tree.rooms[room_id])
            for room_id in self.room_tree.rooms
        }
        # 更新 commands
        if commands is not None and goal is not None:
            self.commands = commands
            self.goal = goal
            return (
                "commands:\n{commands}\n\ngoal:{goal}".format(
                    self.commands, item_id_to_str(self.goal)
                ),
                {}
            )
        random.seed(seed)
        all_items = sorted(list(self.room_tree.itemid_set))
        # use idx to deterministically select goal
        goal = all_items[data_idx % len(all_items)]
        # example: self.goal = "minecraft:dark_oak_sign"
        self.goal = goal
        self.commands = "Enter the room where the target item is located and grasp it."
        return (
            "commands:\n{}\n\ngoal:{}".format(
                self.commands, item_id_to_str(self.goal)
            ),
            {}
        )


    def render(self, mode="human"):
        print(self.step_output())

    def close(self):
        pass
