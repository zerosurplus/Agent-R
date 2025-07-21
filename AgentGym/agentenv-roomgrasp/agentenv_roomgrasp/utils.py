from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ObjectTag:
    """
    表示一个物体的标签，可基于 object_id 或 tag 匹配
    """
    object_id: str = None
    tag: str = None


@dataclass
class ObjectTagWithCount:
    """
    带数量的物体标签
    """
    object_tag: ObjectTag
    count: int


@dataclass
class Furniture:
    furniture_id: str
    furniture_type: str  # "desk", "bed", "cupboard"
    openable: bool = False  # 是否可开合
    opened: bool = False     # 是否已打开
    objects: List[ObjectTagWithCount] = field(default_factory=list)


@dataclass
class FurnitureState:
    opened: bool = False
    objects: List[ObjectTagWithCount] = field(default_factory=list)


@dataclass
class RoomState:
    room_id: str
    door_open: bool = False
    furnitures: dict[str, FurnitureState] = field(default_factory=dict)

@dataclass
class RoomStructure:
    room_id: str
    door_open: bool = False
    furnitures: List[Furniture] = None

    @property
    def furnitures_dict(self):
        return {f.furniture_id: f for f in self.furnitures}


class ActionFailed(Exception):
    pass


def item_id_to_str(item_id: str):
    return item_id.replace("_", " ")
