"""
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""
import math
import random
from copy import deepcopy
import mmengine
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

class MCTSConfig:
    max_depth: int = 10
    iterations: int = 4
    n_generate_samples: int = 4
    coef = 0.25

class MCTSNode:
    def __init__(self, 
                 state=None,
                 parent=None,
                 depth=0,
                 prior=1,
                 llm_response="",
                 env_response="",
                 visits=0,
                 value=0.0,
                 is_terminal=False,
                 mcts_cfg=MCTSConfig()):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = visits
        self.value = value
        self.depth = depth
        self.is_terminal = is_terminal
        self.mcts_cfg = mcts_cfg
        self.llm_response = llm_response
        self.prior = prior
        self.env_response = env_response

    @property
    def is_fully_expanded(self):
        """Check if all possible actions have been expanded into children.
        NOTE: we expand the node once we visit it, therefore if the children is not None,
        this node should be expanded."""
        return len(self.children) > 0
    
    @property
    def reward(self):
        if self.depth > self.mcts_cfg.max_depth:
            return self.mcts_cfg.negative_reward
        else:
            return self.mcts_cfg.positive_reward
        
    @property
    def puct(self) -> float:
        q_value = (self.value / self.visits) if self.visits > 0 else 0
        if self.parent:
            if self.visits == 0:
                u_value = 0
            else:
                u_value = self.mcts_cfg.coef * math.sqrt(math.log(self.parent.visits) / (self.visits))
        else:
            u_value = 0
        return q_value + u_value
    
    def best_child(self):
        """Select the best child node based on UCB1 formula."""
        best_score = float('-inf')
        best_nodes = []
        for child in self.children:
            if child.is_terminal:
                continue
            if child.visits == 0:
                score = float('inf')  # Prioritize unvisited nodes
            else:
                score = child.puct
            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)
        if best_nodes == []:
            self.is_terminal = True
            return None
        return random.choice(best_nodes)

    def update(self, reward):
        """Update the node with the reward from a terminal state."""
        self.visits += 1
        self.value += reward

    def update_terminal(self):
        """Update the node with the reward from a terminal state."""
        is_terminal = True
        for child in self.children:
            if not child.is_terminal:
                is_terminal = False
                break
        self.is_terminal = is_terminal
        

    def to_dict(self):
        return {
            'visits': self.visits,
            'value': self.value,
            'prior': self.prior,
            'puct': self.puct if self.parent is not None else 0,
            'llm_response': self.llm_response,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'children': [child.to_dict() for child in self.children],
            'env_response': self.env_response
        }

class MCTSAgent:
    def __init__(self, 
                agent=None,
                max_iter=10,
                finish_condition=lambda x: x['env_response'] is None,
                generate_cfg=MCTSConfig()):
        self.max_iter = max_iter
        self.finish_condition = finish_condition
        self.agent = agent
        self.generate_cfg = generate_cfg

    def search(self, inputs):
        init_state = ""
        self.root = MCTSNode(state=init_state, llm_response=inputs)

        for iter in range(self.max_iter):
            node = self.root
            if node.is_terminal:
                print(f"Stop at Iter {iter}")
                return
            while node and not node.is_terminal:
                self.expand(node)
                node = self._select(node)
        return
    
    def _generate(self, node):
        agent_response = self.agent.run(node.state)
        is_terminal = self.finish_condition(agent_response)
        # update state for each node
        new_state = deepcopy(node.state)
        new_state.add(role='assistant', content=agent_response['llm_response'])
        if not is_terminal:
            new_state.add(role='tool', content=agent_response['env_response'])
        if agent_response['action']:
            llm_response = str(agent_response['action'])
        else:
            llm_response = agent_response['llm_response']
        new_node = MCTSNode(
            state=new_state,
            parent=node,
            llm_response=llm_response,
            depth=node.depth + 1,
            prior=agent_response['prior'],
            is_terminal=node.depth + 1 > self.generate_cfg.max_depth or is_terminal
        )
        return new_node
    
    def expand(self, node):
        """Expand the node by adding all possible children nodes."""
        if not node.is_fully_expanded:
            # generate sampled actions
            sampled_nodes = [self._generate(node) for _ in range(self.generate_cfg.n_generate_samples)]
            # remove duplicate node
            fingerprint, dedup_nodes = set(), []
            for sample_node in sampled_nodes:
                if sample_node.llm_response in fingerprint:
                    continue
                else:
                    fingerprint.add(sample_node.llm_response)
                    dedup_nodes.append(sample_node)
            node.children = dedup_nodes
            # backpropagate node
            for child in node.children:
                if child.is_terminal:
                    self._backpropagate(child, child.reward)
            
    def _select(self, node):
        """Select the most promising node using UCB1 until an expandable node is reached."""
        while node and node.children:
            node = node.best_child()
        return node

    def _backpropagate(self, node, reward):
        """Propagate the reward back up the tree."""
        while node is not None:
            node.update(reward)
            node.update_terminal()
            node = node.parent

    def save(self, output_path):
        state_dict = self.root.to_dict()
        mmengine.dump(state_dict, output_path, ensure_ascii=False, indent=2)

    def _load(self, data_path):
        self.root = self.load(data_path)

    @staticmethod
    def load(data_path):
        state_dict = mmengine.load(data_path)

        def dict_to_node(data):
            # Recursively construct nodes from dictionary data
            children_data = data.pop('children')
            node = MCTSNode(**data)
            node.children = [dict_to_node(child) for child in children_data]
            return node

        return dict_to_node(state_dict)
