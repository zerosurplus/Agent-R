from dataclasses import dataclass
from typing import Any, Mapping
import re
import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseTask, BaseEnvClient, ConversationMessage, StepOutput


@dataclass
class RoomGraspEnvClient(BaseEnvClient):
    """
    房间抓取任务的客户端接口
    支持通过 HTTP 接口与环境交互
    """

    conversation_start = (
        ConversationMessage({
            "from": "human",
            "loss": None,
            "value": 'You are in an environment composed of multiple rooms (including bathroom, bedroom, kitchen, living_room, and study_room). Your task is to grasp the target item specified below.\n\n### Available Actions:\n- `move <room name or furniture name>`: Move to a room door or next to a piece of furniture.\n- `open <room name or furniture name>`: Open a closed room door or openable furniture (e.g., cupboard).\n- `enter <room name>`: Enter an already opened room.\n- `exit`: Exit the current room and return to the outside.\n- `grasp <item name>`: Grasp an item currently accessible in the room or opened furniture.\n- `inventory`: Check your inventory.\n\n### Rules:\n1. You must **open a room\'s door** before entering it.\n2. You can only **move to furniture** inside a room after entering it.\n3. Some furniture (e.g., cupboards) may need to be **opened** before accessing its contents.\n4. You cannot directly move between rooms—you must first `exit` the current room.\n5. If an action fails, carefully check your location and state before retrying.\n\n### Instruction:\nAt each step, I will provide an observation.\nYou must respond **only** with `Thought:` and `Action:` using **exactly** these commands: `move`, `open`, `enter`, `exit`, `grasp`, or `inventory`.\n\n### Response Format:\nThought:\n[Your reasoning]\n\nAction:\n[One valid command]'
        }),
        ConversationMessage({
            "from": "gpt",
            "loss": False,
            "value": "OK. I'll follow your instructions and try my best to solve the task."
        }),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        room_config_dir: str = "agentenv_roomgrasp/",
        commands: str = None,
        goal: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        dir_info = {"room_config_dir": room_config_dir, "commands": commands, "goal": goal}
        ok = requests.post(f"{self.env_server_base}/create", json=dir_info, timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")
        
        ok = ok.json()
        print(ok)
        self.env_id = ok["id"]
        self.info = {
            "observation": ok["observation"],
            "reward": 0,
            "done": False,
        }

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        # 预处理动作字符串
        action = action.split("Instruction:")[0].split("Action:")[-1]
        action = re.sub(r"[^A-Za-z0-9, _]+", "", action).strip()

        response = self._post("step", {"action": action})
        print(response)
        self.info.update({
            "observation": response.get("observation", ""),
            "reward": response.get("reward", 0),
            "done": response.get("done", False),
        })
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": idx})
        self.info.update({
            "observation": response.get("observation", ""),
            "reward": 0,
            "done": False,
        })
        return response


@dataclass
class RoomGraspTask(BaseTask):
    """
    房间抓取任务定义
    """
    env_client_cls = RoomGraspEnvClient
    env_name = "RoomGrasp"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int = 1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)