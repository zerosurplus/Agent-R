import requests
import unittest
from agentenv_roomgrasp.room_tree import RoomTree

ENV_SERVER = "http://localhost:36001"


class TestRoomGraspEnv(unittest.TestCase):
    def setUp(self):
        self.env_id = self._create_env()

    def _create_env(self, goal="red_cube"):
        resp = requests.post(f"{ENV_SERVER}/create", json={"goal": goal})
        self.assertEqual(resp.status_code, 200, f"Create failed: {resp.text}")
        data = resp.json()
        print("✅ Created env with goal:", data.get("goal"))
        self.assertIn("id", data, f"Create env failed with error: {data}")
        return data["id"]

    def _step(self, env_id, action, expected_obs=None):
        step_resp = requests.post(f"{ENV_SERVER}/step", json={"id": env_id, "action": action})
        self.assertEqual(step_resp.status_code, 200)
        data = step_resp.json()
        if expected_obs:
            self.assertIn(expected_obs, data["observation"], f"Observation mismatch: {data['observation']}")
        return data

    def _get_observation(self, env_id):
        obs_resp = requests.get(f"{ENV_SERVER}/observation", params={"id": env_id})
        
        try:
            # 尝试解析为 JSON
            data = obs_resp.json()
        except requests.exceptions.JSONDecodeError:
            # 如果失败，将原始文本包装成统一格式的 dict
            data = {"observation": obs_resp.text}

        # 确保 data 是字典，并且包含 observation 字段
        if isinstance(data, dict):
            if "error" in data:
                raise ValueError(f"Server returned error: {data['error']}")
            self.assertIn("observation", data, f"Observation field missing in response: {data}")
            return data["observation"]
        else:
            raise ValueError(f"Unexpected response type: {type(data)}, raw content: {obs_resp.text}")

    def test_01_create_env(self):
        """测试创建环境"""
        env_id = self._create_env()
        self.assertIsInstance(env_id, int)
        print("✅ 环境创建成功:", env_id)

    def test_02_step_move(self):
        """测试移动动作"""
        env_id = self._create_env()
        data = self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        self.assertIn("reward", data)
        self.assertIn("done", data)
        print("✅ 移动动作成功:", data)

    def test_03_step_open(self):
        """测试开门动作"""
        env_id = self._create_env()
        self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        data = self._step(env_id, "Open living_room", "You opened the door of living_room")
        print("✅ 开门动作成功:", data)

    def test_04_grasp_item_with_enter(self):
        """测试进入房间后抓取物品"""
        env_id = self._create_env(goal="red_cube")
        self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        self._step(env_id, "Open living_room", "You opened the door of living_room")
        self._step(env_id, "Enter living_room", "You entered living_room")
        data = self._step(env_id, "Grasp red_cube", "You grasped red cube")
        self.assertEqual(data["reward"], 1.0)
        self.assertTrue(data["done"])

    def test_05_move_to_desk(self):
        """测试移动到家具旁（如 desk_01）"""
        env_id = self._create_env()
        self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        self._step(env_id, "Open living_room", "You opened the door of living_room")
        self._step(env_id, "Enter living_room", "You entered living_room")
        data = self._step(env_id, "Move desk_01", "You arrived next to desk_01")
        print("✅ 成功移动到桌子旁:", data)

    def test_06_grasp_from_desk_with_enter(self):
        """测试从桌子旁抓取物品，先 Enter 房间"""
        env_id = self._create_env(goal="red_cube")
        self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        self._step(env_id, "Open living_room", "You opened the door of living_room")
        self._step(env_id, "Enter living_room", "You entered living_room")
        self._step(env_id, "Move desk_01", "You arrived next to desk_01")
        data = self._step(env_id, "Grasp red_cube", "You grasped red cube")
        self.assertEqual(data["reward"], 1.0)
        self.assertTrue(data["done"])

    def test_07_full_episode_with_enter(self):
        """完整流程：Move → Open → Enter → Grasp"""
        env_id = self._create_env()
        actions = [
            ("Move living_room", "You arrived at the door of living_room"),
            ("Open living_room", "You opened the door of living_room"),
            ("Enter living_room", "You entered living_room"),
            ("Grasp red_cube", "You grasped red cube"),
        ]

        for action, expected in actions:
            with self.subTest(action=action):
                data = self._step(env_id, action, expected)
                self.assertIn(expected, data["observation"])

        print("✅ 完整动作序列测试通过")

    def test_08_open_cupboard_and_grasp_with_enter(self):
        """打开柜子并抓取物品，包含 Enter 步骤"""
        env_id = self._create_env(goal="blue_sphere")
        self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        self._step(env_id, "Open living_room", "You opened the door of living_room")
        self._step(env_id, "Enter living_room", "You entered living_room")
        self._step(env_id, "Move cupboard_01", "You arrived next to cupboard_01")
        self._step(env_id, "Open cupboard_01", "You opened cupboard_01")
        data = self._step(env_id, "Grasp blue_sphere", "You grasped blue sphere")
        self.assertEqual(data["reward"], 1.0)
        self.assertTrue(data["done"])

    def test_09_open_invalid_furniture(self):
        """测试打开无效或不可开合的家具"""
        env_id = self._create_env()
        self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        self._step(env_id, "Open living_room", "You opened the door of living_room")
        self._step(env_id, "Enter living_room", "You entered living_room")

        # 尝试打开不存在的家具
        data = self._step(env_id, "Open non_existent", "Cannot open non_existent, it is not a room or furniture")
        self.assertIn("not a room or furniture", data["observation"])

        # 尝试打开非 openable 的家具
        self._step(env_id, "Move desk_01", "You arrived next to desk_01")
        data = self._step(env_id, "Open desk_01", "desk_01 cannot be opened")
        self.assertIn("cannot be opened", data["observation"])

        print("✅ 异常家具操作测试通过")

    def test_10_move_to_cupboard(self):
        """测试移动到家具旁（如 cupboard_01）"""
        env_id = self._create_env()
        self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        self._step(env_id, "Open living_room", "You opened the door of living_room")
        self._step(env_id, "Enter living_room", "You entered living_room")
        data = self._step(env_id, "Move cupboard_01", "You arrived next to cupboard_01")
        print("✅ 成功移动到家具旁:", data)

    def test_11_grasp_from_closed_cupboard(self):
        """测试从关闭的柜子中抓取物品是否失败"""
        env_id = self._create_env(goal="blue_sphere")
        self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        self._step(env_id, "Open living_room", "You opened the door of living_room")
        self._step(env_id, "Enter living_room", "You entered living_room")
        self._step(env_id, "Move cupboard_01", "You arrived next to cupboard_01")

        # 不打开 cupboard 直接尝试抓取
        data = self._step(env_id, "Grasp blue_sphere", "cupboard_01 is closed, please open it first")
        print("✅ 成功阻止从关闭的柜子中取物")

    def test_12_check_room_state_after_grasp_with_enter(self):
        """验证房间状态更新"""
        env_id = self._create_env(goal="blue_sphere")
        self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        self._step(env_id, "Open living_room", "You opened the door of living_room")
        self._step(env_id, "Enter living_room", "You entered living_room")
        self._step(env_id, "Move cupboard_01", "You arrived next to cupboard_01")
        self._step(env_id, "Open cupboard_01", "You opened cupboard_01")
        obs_before = self._step(env_id, "inventory", "empty")["observation"]
        self.assertNotIn("blue_sphere", obs_before)  # ✅ 改为断言不存在

        data = self._step(env_id, "Grasp blue_sphere", "You grasped blue sphere")

        obs_after = self._step(env_id, "inventory", "Your inventory contains: blue sphere")["observation"]
        self.assertIn("blue sphere", obs_after)  # ✅ 抓取后才出现在背包中
        self.assertEqual(data["reward"], 1.0)
        self.assertTrue(data["done"])


    def test_13_invalid_room(self):
        """测试访问不存在的房间"""
        env_id = self._create_env()
        data = self._step(env_id, "Move unknown_room", "Cannot move to unknown_room")
        self.assertEqual(data["reward"], 0.0)
        self.assertFalse(data["done"])

    def test_14_tag_based_item_matching(self):
        """测试基于 tag 的物品匹配（如 food 类）"""
        env_id = self._create_env(goal="food")
        self._step(env_id, "Move kitchen", "You arrived at the door of kitchen")
        self._step(env_id, "Open kitchen", "You opened the door of kitchen")
        self._step(env_id, "Enter kitchen", "You entered kitchen")
        self._step(env_id, "Move fridge", "You arrived next to fridge")
        self._step(env_id, "Open fridge", "You opened fridge")
        data = self._step(env_id, "Grasp apple", "You grasped apple")
        self.assertEqual(data["reward"], 1.0)
        self.assertTrue(data["done"])

    def test_15_inter_room_grasp(self):
        """测试从当前房间尝试抓取其他房间的物品"""
        env_id = self._create_env(goal="apple")
        self._step(env_id, "Move living_room", "You arrived at the door of living_room")
        self._step(env_id, "Open living_room", "You opened the door of living_room")
        self._step(env_id, "Enter living_room", "You entered living_room")
        data = self._step(env_id, "Grasp apple", "living_room does not contain apple")
        self.assertEqual(data["reward"], 0.0)
        self.assertFalse(data["done"])

    def test_16_switch_furniture_with_enter(self):
        """同一个房间切换家具"""
        env_id = self._create_env(goal="shirt")
        self._step(env_id, "Move bedroom", "You arrived at the door of bedroom")
        self._step(env_id, "Open bedroom", "You opened the door of bedroom")
        self._step(env_id, "Enter bedroom", "You entered bedroom")

        # 直接切换家具
        self._step(env_id, "Move wardrobe", "You arrived next to wardrobe")
        self._step(env_id, "Move nightstand", "You arrived next to nightstand")  # ✅ 应该成功
        data = self._step(env_id, "Grasp key", "You grasped key")
        self.assertIn("key", data["observation"])

    def test_17_invalid_room_name_with_underscore(self):
        """测试带有下划线的无效房间名（如 study_room）"""
        env_id = self._create_env()
        data = self._step(env_id, "Move study_room", "You arrived at the door of study_room")
        self.assertEqual(data["reward"], 0.0)
        self.assertFalse(data["done"])


if __name__ == "__main__":
    unittest.main()