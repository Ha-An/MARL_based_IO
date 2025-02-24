import multiprocessing as mp


def worker_process(env_fn, conn):
    """
    자식 프로세스에서 돌릴 함수:
    - env_fn()으로 환경 하나 생성
    - 메인 프로세스로부터 명령을 받아 step/ reset 처리
    """
    env = env_fn()
    while True:
        cmd, data = conn.recv()  # (command, payload)
        if cmd == 'reset':
            obs = env.reset()
            conn.send(obs)
        elif cmd == 'step':
            actions = data
            obs, reward, done, info = env.step(actions)
            conn.send((obs, reward, done, info))
        elif cmd == 'close':
            conn.close()
            break
        else:
            raise NotImplementedError


class ParallelEnvManager:
    def __init__(self, env_fn_list):
        """
        env_fn_list: [lambda: InventoryManagementEnv(), lambda:..., ...] (N개)
        """
        self.n_envs = len(env_fn_list)
        self.parent_conns = []
        self.processes = []

        for fn in env_fn_list:
            parent_conn, child_conn = mp.Pipe()
            proc = mp.Process(target=worker_process, args=(fn, child_conn))
            proc.start()
            self.parent_conns.append(parent_conn)
            self.processes.append(proc)

    def reset(self):
        obs_list = []
        for conn in self.parent_conns:
            conn.send(('reset', None))
        for conn in self.parent_conns:
            obs = conn.recv()
            obs_list.append(obs)
        return obs_list

    def step(self, actions):
        """
        actions: 병렬 환경 각각의 action (MultiDiscrete여도, 여기선 단순 리스트로 가정)
        """
        for conn, act in zip(self.parent_conns, actions):
            conn.send(('step', act))
        results = []
        for conn in self.parent_conns:
            results.append(conn.recv())
        obs_list, reward_list, done_list, info_list = zip(*results)
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    def close(self):
        for conn in self.parent_conns:
            conn.send(('close', None))
        for p in self.processes:
            p.join()
