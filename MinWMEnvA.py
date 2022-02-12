import gym
import numpy as np


class MinWMEnvA(gym.Env):
    def __init__(self, config):
        self.action_space = gym.spaces.Discrete(4)
        self.dim = config["dim"]
        self.obs_dim = self.dim * 3 + 2
        self.observation_space = gym.spaces.Box(low=np.zeros(self.obs_dim, dtype=np.int),
                                                high=np.ones(self.obs_dim, dtype='int'))
        self.task_switch = 0
        self.sample = np.zeros(self.dim, dtype='int')
        self.target = np.zeros(self.dim, dtype='int')
        self.control = np.array([0, 0])
        self.sample_period = config["sample_period"]
        self.response_period = config["response_period"]
        self.match_delay = config["match_delay"]
        self.reward_delay = config["reward_delay"]
        self.action = 0
        self.done = False
        self.count = 0

    def reset(self):
        self.action = 0
        self.done = False
        self.count = 0
        self.task_switch = np.random.randint(0, self.dim, dtype='int')
        self.sample = np.random.randint(0, 2, self.dim, dtype='int')
        self.target = np.random.randint(0, 2, self.dim, dtype='int')
        return np.zeros(self.obs_dim, dtype=np.int)

    def step(self, action):
        reward = 0
        self.count += 1
        observation = np.zeros(self.obs_dim, dtype=np.int)
        task_switch = np.zeros(self.dim, dtype='int')
        if self.count <= self.sample_period + 1:
            # showing the sample
            sample = np.zeros(self.dim * 2, dtype='int')
            for i in range(self.dim):
                if i == self.task_switch:
                    task_switch[i] = 1
                if self.sample[i] == 0:
                    sample[i*2] = 1
                    sample[i*2+1] = 0
                else:
                    sample[i*2] = 0
                    sample[i*2+1] = 1
            control = np.array([1, 0])
            observation = np.append(task_switch, np.append(sample, control))
        elif self.count <= self.sample_period + self.match_delay + 1:   # match delay
            pass
        elif self.count <= self.sample_period + self.match_delay + self.response_period + 1:
            # response period
            target = np.zeros(self.dim * 2, dtype='int')
            for i in range(self.dim):
                if self.target[i] == 0:
                    target[i*2] = 1
                    target[i*2+1] = 0
                else:
                    target[i*2] = 0
                    target[i*2+1] = 1
            control = np.array([1, 1])
            observation = np.append(task_switch, np.append(target, control))
            if action > 0 and self.count > self.sample_period + self.match_delay + 2:
                self.action = action    # response
        elif self.count <= self.sample_period + self.match_delay + self.response_period + self.reward_delay + 1:
            pass    # reward delay
        else:
            reward = 0.0
            if self.sample[self.task_switch] == self.target[self.task_switch] and self.action == 2:
                reward = 1.0
            if self.sample[self.task_switch] != self.target[self.task_switch] and self.action == 1:
                reward = 1.0
            self.done = True
        return observation, reward, self.done, {}

    def render(self):
        pass


def main():
    config = {"dim": 3, "sample_period": 2, "response_period": 3, "match_delay": 1, "reward_delay": 1}
    env = MinWMEnvA(config)
    for i in range(50):
        obs = env.reset()
        while True:
            control = obs[config["dim"] * 3:]
            if np.array_equal(control, np.array([1, 1])):
                action = np.random.randint(0, 4)
                print(obs, action)  # , end=",")
            else:
                action = 0
            obs, reward, done, info = env.step(action)
            if done:
                print("reward:", reward)
                break


if __name__ == '__main__':
    main()
