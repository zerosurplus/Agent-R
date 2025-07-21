from typing import Dict, Any
from dataclasses import dataclass

from .environment import RoomGraspEnv
from .room_tree import RoomTree
import os


class RoomGrasp_Wrapper:
    """
    环境封装器，支持多个环境实例同时运行。
    提供 create / step / reset / get_observation 接口。
    """

    def __init__(self, room_config_dir="agentenv_roomgrasp/"):
        self._max_id = 0
        self.env = {}  # dict[id, env_item]
        self.info = {}  # dict[id, env_info]
        self.room_tree = RoomTree(room_config_dir=room_config_dir)

    def create(self, commands: str = None, goal: str = None):
        try:
            id = self._max_id
            new_env = RoomGraspEnv(
                room_tree=self.room_tree, commands=commands, goal=goal
            )
            ob, _ = new_env.reset(data_idx=id)
            print(f"-------Env {id} created--------")
            payload = {"id": id, "observation": ob, "done": False, "reward": 0}
            self.env[id] = new_env
            self.info[id] = {
                "observation": ob,
                "done": False,
                "reward": 0,
                "deleted": False,
            }
            self._max_id += 1
        except Exception as e:
            import traceback
            print("[ERROR]", traceback.format_exc())
            payload = {"error": f"{e}"}
        return payload

    def step(self, id: int, action: str):
        try:
            self._check_id(id)
            print(f"[DEBUG] Action before step: {action}")
            (ob, reward, done, _, _) = self.env[id].step(action)
            payload = {"observation": ob, "reward": reward, "done": done}
            self.info[id].update(payload)
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload

    def reset(self, id: int, data_idx: int):
        try:
            self._check_id(id)
            ob, _ = self.env[id].reset(data_idx=data_idx)
            payload = {"id": id, "observation": ob, "done": False, "reward": 0}
            self.info[id].update(
                {"observation": ob, "done": False, "reward": 0, "deleted": False}
            )
        except Exception as e:
            payload = {"error": str(e)}
        return payload

    def get_observation(self, id: int):
        try:
            self._check_id(id)
            return self.info[id]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_info(self, id: int):
        try:
            self._check_id(id)
            return self.info[id]
        except Exception as e:
            return {"error": str(e)}

    def _check_id(self, id: int):
        if id not in self.info:
            raise NameError(f"The id {id} is not valid.")
        if self.info[id]["deleted"]:
            raise NameError(f"The task with environment {id} has been deleted.")

server = RoomGrasp_Wrapper()
