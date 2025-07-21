# roomgraspenv/room_tree.py
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import os
from .utils import ObjectTag, ObjectTagWithCount, Furniture, RoomStructure, RoomState, FurnitureState

class RoomTree:
    """
    房间结构树解析器，从 JSON 加载房间配置。
    类似 crafting_tree.py 的功能。
    """

    def __init__(self, room_config_dir: str):
        self.room_data = {}       # 存储原始房间数据
        self.rooms = {}           # 房间 ID → RoomStructure
        self.roomid_set = set()   # 所有房间ID集合
        self.tag_set = set()      # 标签集合（tag:xxx）
        self.itemid_set = set()   # 所有物品ID集合
        self.item_id_to_tag = {}  # 物品ID → 标签映射
        self.furniture_set = set() # 家具ID集合

        self._load_rooms(room_config_dir)  # 加载房间文件
        self.clean_up_rooms()            # 清理不一致或无效数据
        
    def _load_rooms(self, room_config_dir: str):
        for filename in os.listdir(os.path.join(room_config_dir, "data", "rooms")):
            print("Loading rooms from:", os.path.join(room_config_dir, "data", "rooms"))  # 调试输出
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(room_config_dir, "data", "rooms", filename)

            try:
                with open(file_path, "r") as f:
                    room_data = json.load(f)
                    print(f"[DEBUG] Loaded room from {filename}: {room_data}")
                    self.room_data[room_data["id"]] = room_data
            except Exception as e:
                print(f"[ERROR] Failed to load {file_path}: {e}")
                continue

            room_id = room_data["id"]
            self.roomid_set.add(room_id)

            furnitures = []

            if "furnitures" in room_data:
                for furniture_data in room_data["furnitures"]:
                    furniture_id = furniture_data.get("furniture_id")
                    print(f"[DEBUG] Loading furniture: {furniture_id} in room {room_id}")  # 调试输出
                    furniture_type = furniture_data.get("furniture_type", "desk")
                    openable = furniture_data.get("openable", False)
                    opened = furniture_data.get("opened", False)

                    furniture = Furniture(
                        furniture_id=furniture_id,
                        furniture_type=furniture_type,
                        openable=openable,
                        opened=opened,
                        objects=[]
                    )

                    # 解析家具中的物品
                    for obj in furniture_data.get("objects", []):
                        if isinstance(obj, dict):
                            object_id = obj.get("id")
                            tag = obj.get("tag")
                            count = obj.get("count", 1)
                            if object_id:
                                furniture.objects.append(
                                    ObjectTagWithCount(ObjectTag(object_id=object_id), count)
                                )
                                self.itemid_set.add(object_id)
                                self.item_id_to_tag[object_id] = tag
                            elif tag:
                                furniture.objects.append(
                                    ObjectTagWithCount(ObjectTag(tag=tag), count)
                                )
                                self.tag_set.add(tag)
                        else:
                            furniture.objects.append(
                                ObjectTagWithCount(ObjectTag(object_id=obj), 1)
                            )
                            self.itemid_set.add(obj)

                    furnitures.append(furniture)
                    self.furniture_set.add(furniture_id)

            self.rooms[room_id] = RoomStructure(
                room_id=room_id,
                door_open=room_data.get("door_open", False),
                furnitures=furnitures
            )

    def clean_up_rooms(self):
        # 移除没有家具的房间（可选）
        valid_rooms = {}
        for room_id, room in self.rooms.items():
            if room.furnitures:
                valid_rooms[room_id] = room
            else:
                print(f"[INFO] Removed empty room: {room_id}")

        self.rooms = valid_rooms

    def is_valid_item(self, item_name: str) -> bool:
        return item_name in self.itemid_set
    
    def is_valid_room(self, room_name: str) -> bool:
        return room_name in self.rooms

    def get_items_with_tag(self, tag: str) -> List[str]:
        return [item for item, t in self.item_id_to_tag.items() if t == tag]

    def create_room_set(self, goal_room: str) -> List[str]:
        rooms = list(self.rooms.keys())
        if goal_room not in rooms:
            raise ValueError(f"Goal room {goal_room} not found in config.")
        return [goal_room] + [r for r in rooms if r != goal_room]

    # room_tree.py

    def _create_room_state(self, room_structure):
        """
        根据 RoomStructure 创建初始 RoomState
        """
        return RoomState(
            room_id=room_structure.room_id,
            door_open=room_structure.door_open,
            furnitures={
                furniture.furniture_id: FurnitureState(
                    opened=furniture.opened,
                    objects=list(furniture.objects)  # 拷贝原始物品列表
                )
                for furniture in room_structure.furnitures
            },
        )
    
    def is_item_in_tag(self, tag: str, item_id: str) -> bool:
        return item_id in self.get_items_with_tag(tag)