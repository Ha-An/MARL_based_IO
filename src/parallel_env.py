import concurrent.futures
import numpy as np

# 기존에 작성된 InventoryManagementEnv를 import
from GymEnvironment import InventoryManagementEnv


class ParallelInventoryManagementEnv:
    """
    여러 개의 InventoryManagementEnv를 동시에 구동하기 위한 간단한 병렬 환경 래퍼.
    ThreadPoolExecutor를 사용하였으나, 실제 CPU 부하가 큰 경우 ProcessPoolExecutor로 변경하는 것도 방법.
    """

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [InventoryManagementEnv() for _ in range(num_envs)]
        # 주의: 각 Env에서 SimPy를 어떻게 초기화하고 reset하는지에 따라
        #       병렬 실행 시 충돌이 없도록 확인해야 합니다.

    def reset(self):
        """
        모든 환경을 초기화하여 초기 state를 반환.
        returns: 초기 관측값들의 리스트. 길이는 self.num_envs
        """
        def _reset(env):
            return env.reset()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            results = list(executor.map(_reset, self.envs))
        # results: [ state_env0, state_env1, ..., state_envN ]
        return results

    def step(self, actions):
        """
        모든 환경에 대해 각기 다른 action을 적용하고 step을 진행.
        actions: 길이가 self.num_envs인 리스트(각 원소는 [num_agents 길이의 action 배열])
        returns: 각 환경별 (obs, reward, done, info) 튜플을 모아서 반환
        """
        if len(actions) != self.num_envs:
            raise ValueError(
                f"actions의 길이({len(actions)})가 num_envs({self.num_envs})와 일치해야 합니다.")

        def _step(args):
            env, act = args
            return env.step(act)

        # 병렬로 step 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            results = list(executor.map(_step, zip(self.envs, actions)))
        # results: [(obs_env0, reward_env0, done_env0, info_env0),
        #           (obs_env1, reward_env1, done_env1, info_env1),
        #           ...
        #           (obs_envN, reward_envN, done_envN, info_envN)]

        # 관례적으로 vector-env는 obs, reward, done, info를 튜플로 묶어서 반환
        obs_list, reward_list, done_list, info_list = zip(*results)

        return obs_list, reward_list, done_list, info_list

    def close(self):
        """
        자원 정리용 (필요 시 사용).
        """
        pass
