from pydantic import BaseModel
from typing import Optional

class CreateRequestBody(BaseModel):
    
    goal: Optional[str] = None  # 目标，如 "房间A中的红色方块"


class StepRequestBody(BaseModel):
    id: int      # 环境 ID
    action: str  # 动作字符串，如 "Move 房间A"


class ResetRequestBody(BaseModel):
    id: int
    data_idx: Optional[int] = 0
