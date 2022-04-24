#!/usr/bin/env python

import sys
import os
import argparse
import json

import numpy as np
import gym

import brica1.brica_gym
import brical

import CBT1cCA_1

class Register(brica1.brica_gym.Component):
    def __init__(self, dim, divisor, persistence):
        super().__init__()
        self.make_in_port('in', dim)
        self.make_in_port('attention', divisor)   # the section to be held
        self.make_in_port('token_in', 1)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('out', dim + 2)
        self.make_out_port('reward', 1)
        self.make_out_port('done', 1)
        self.make_out_port('token_out', 1)
        self.dim = dim
        self.divisor = divisor
        self.persistence = persistence  # time for holding the state
        self.state = np.zeros(dim, dtype=np.int)
        self.counter = 0
        self.token = 0
        self.prev_in = np.zeros(dim, dtype=np.int)
        self.position = -1
        self.recognized = np.array([0, 0])

    def fire(self):
        in_data = self.get_in_port('in').buffer
        self.recognized = np.array([0, 0])
        attention = self.get_in_port('attention').buffer
        if np.max(attention) < 1: # no command for holding
            if np.max(self.state) > 0:    # holding
                self.counter += 1
                if self.position >= 0:
                    span = self.dim // self.divisor
                    in_data_spot = in_data[span * self.position:span * (self.position + 1)]
                    if np.max(in_data_spot) > 0 and np.min(in_data_spot) < 1:   # excluding control and task switch
                        if np.array_equal(self.state[span * self.position:span * (self.position + 1)],
                                          in_data_spot):
                            self.recognized = np.array([0, 1])
                        else:
                            self.recognized = np.array([1, 0])
        else: # the section to be held
            if self.counter == 0:
                span = self.dim // self.divisor
                self.position = np.argmax(attention)
                self.state = np.zeros(self.dim, dtype=np.int)
                self.state[span * self.position:span * (self.position + 1)] \
                    = in_data[span * self.position:span * (self.position + 1)]
                self.counter = 1
            elif np.max(self.state) > 0:   # rehold the state
                self.counter = 1
            else:
                self.counter = 0
        if  self.counter >= self.persistence:
            self.counter = 0
            self.state = np.zeros(self.dim, dtype=np.int)
        self.token = self.inputs['token_in'][0]
        self.prev_in = in_data
        out = np.concatenate([in_data, self.recognized])
        # out[:self.dim-2] = np.zeros(self.dim-2, dtype=np.int)
        self.results['out'] = out
        self.results['reward'] = self.inputs['reward']
        self.results['done'] = self.inputs['done']

    def reset(self):
        self.token = 0
        self.counter = 0
        self.inputs['in'] = np.zeros(self.dim, dtype=np.int)
        self.state = np.zeros(self.dim, dtype=np.int)
        self.inputs['attention'] = np.zeros(self.divisor, dtype=np.int)
        self.results['out'] = np.zeros(self.dim + 2, dtype=np.int)
        self.results['token_out'] = np.array([0])
        self.results['done'] = np.array([0])
        self.results['reward'] = np.array([0.0])
        self.get_out_port('token_out').buffer = self.results['token_out']
        self.get_out_port('done').buffer = self.results['done']
        self.get_out_port('reward').buffer = self.results['reward']
        self.prev_in = np.zeros(self.dim, dtype=np.int)
        self.position = -1
        self.recognized = np.array([0, 0])


class RCStub(brica1.brica_gym.Component):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config['in_dim']
        self.dim = (self.in_dim-2) // 3
        self.n_action = config['n_action']  # number of action choices
        self.make_in_port('observation', self.in_dim)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('action', self.n_action)
        self.make_in_port('token_in', 1)
        self.make_out_port('token_out', 1)
        self.make_out_port('done', 1)
        self.token = 0
        self.gone = False

    def fire(self):
        in_data = self.get_in_port('observation').buffer
        self.results['action'] = np.zeros(self.n_action, dtype=np.int)
        if in_data[self.in_dim-2]==1 and in_data[self.in_dim-1]==0 and not self.gone: # sample period
            task_switch_dim = self.dim//2   # dim should be odd
            self.results['action'][task_switch_dim:task_switch_dim+self.dim] = in_data[:self.dim]
            self.gone = True
        self.token = self.inputs['token_in'][0]
        self.results['done'] = self.inputs['done']

    def reset(self):
        self.token = 0
        self.init = True
        self.gone = False
        self.inputs['token_in'] = np.array([0])
        self.inputs['reward'] = np.array([0.0])
        self.results['token_out'] = np.array([0])
        self.results['done'] = np.array([0])
        self.results['reward'] = np.array([0.0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_in_port('reward').buffer = self.inputs['reward']
        self.get_out_port('token_out').buffer = self.results['token_out']
        self.get_out_port('done').buffer = self.results['done']

    def close(self):
        pass


class ADStub(brica1.brica_gym.Component):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config['in_dim']
        self.n_action = config['n_action']    # number of action choices
        self.dim = self.in_dim - 1
        self.make_in_port('observation', self.in_dim)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('action', self.n_action)
        self.make_in_port('token_in', 1)
        self.make_out_port('token_out', 1)
        self.make_out_port('done', 1)
        self.token = 0
        self.prev_actions = 0
        self.init = True
        self.gone = False

    def fire(self):
        self.results['action'] = np.array([0, 0])
        in_data = self.get_in_port('observation').buffer
        if in_data[-4] == 1 and in_data[-3] == 1 and not self.gone:  # response period
            if np.max(in_data[-2:]) > 0:
                self.results['action'] = in_data[-2:]
                self.gone = True
        self.token = self.inputs['token_in'][0]
        self.results['done'] = self.inputs['done']

    def reset(self):
        self.token = 0
        self.init = True
        self.inputs['token_in'] = np.array([0])
        self.inputs['reward'] = np.array([0.0])
        self.results['token_out'] = np.array([0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_in_port('reward').buffer = self.inputs['reward']
        self.get_out_port('token_out').buffer = self.results['token_out']
        self.gone = False

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description='BriCA Minimal Cognitive Architecture with Gym')
    parser.add_argument('--dump', help='dump file path')
    parser.add_argument('--episode_count', type=int, default=1, metavar='N',
                        help='Number of training episodes (default: 1)')
    parser.add_argument('--max_steps', type=int, default=30, metavar='N',
                        help='Max steps in an episode (default: 20)')
    parser.add_argument('--config', type=str, default='MinWMAgent1.json', metavar='N',
                        help='Model configuration (default: MinWMAgent1.json')
    parser.add_argument('--dump_flags', type=str, default="",
                        help='a:all, s:skim')
    parser.add_argument('--brical', type=str, default='MinWMAgent1.brical.json', metavar='N',
                        help='a BriCAL json file')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    nb = brical.NetworkBuilder()
    f = open(args.brical)
    nb.load_file(f)
    if not nb.check_consistency():
        sys.stderr.write("ERROR: " + args.brical + " is not consistent!\n")
        exit(-1)

    if not nb.check_grounding():
        sys.stderr.write("ERROR: " + args.brical + " is not grounded!\n")
        exit(-1)

    train = {"episode_count": args.episode_count, "max_steps": args.max_steps, "dump_flags": args.dump_flags}
    config['train'] = train

    if args.dump is not None and args.dump_flags != "":
        try:
            dump = open(args.dump, mode='w')
        except IOError:
            print('Dump path error', file=sys.stderr)
            sys.exit(1)
    else:
        dump = None
    train["dump"] = dump
    config["ActionDeterminer"]["NeoCortex"]["ActionPredictor"]["dump"] = dump
    config["RegisterUnit"]["NeoCortex"]["ActionPredictor"]["dump"] = dump

    env = gym.make(config['env']['name'], config=config['env'])

    nb.unit_dic['MinWMAgent1.ActionDeterminer'].__init__(config['ActionDeterminer']['learning_mode'],
                                                         config['train'], config['ActionDeterminer'])
    nb.unit_dic['MinWMAgent1.RegisterController'].__init__(config['RegisterUnit']['learning_mode'], train,
                                                           config['RegisterUnit'])
    nb.unit_dic['MinWMAgent1.Register'].__init__(config['RegisterUnit']['in_dim'], config['RegisterUnit']['n_action'],
                                                 config['RegisterUnit']['persistence'])

    nb.make_ports()

    agent_builder = brical.AgentBuilder()
    model = nb.unit_dic['MinWMAgent1.MinWMAgent']
    agent = agent_builder.create_gym_agent(nb, model, env)
    scheduler = brica1.VirtualTimeSyncScheduler(agent)

    dump_cycle = config["dump_cycle"]
    dump_counter = 0
    reward_sum = 0.0
    ad_gone_counter = 0
    rc_gone_counter = 0
    rc_sample_gone = 0
    correct_wm_cnt = 0
    ad_target_gone = 0
    ad_correct_cnt = 0
    for i in range(train["episode_count"]):
        last_token = 0
        correct_wm = False
        for j in range(train["max_steps"]):
            scheduler.step()
            current_token = agent.get_out_port('token_out').buffer[0]
            if last_token + 1 == current_token:
                last_token = current_token
                if "a" in train["dump_flags"]:
                    dump.write("{0}:{1},{2},{3},{4},{5}\n".
                                   format(agent.get_out_port("token_out").buffer[0],
                                          str(agent.get_in_port("observation").buffer.tolist()),
                                          str(nb.unit_dic['MinWMAgent1.Register'].inputs['attention']),
                                          str(nb.unit_dic['MinWMAgent1.Register'].state),
                                          nb.unit_dic['MinWMAgent1.Register'].recognized,
                                          agent.get_out_port("action").buffer))
                if np.array_equal(nb.unit_dic['MinWMAgent1.Register'].inputs["in"][-2:], [1, 0]) and \
                        max(nb.unit_dic['MinWMAgent1.RegisterController'].get_out_port("action").buffer) > 0:
                    rc_sample_gone += 1
                    if np.array_equal(nb.unit_dic['MinWMAgent1.Register'].inputs["in"][:env.env.dim],
                                      nb.unit_dic['MinWMAgent1.RegisterController'].get_out_port("action").
                                              buffer[env.env.dim//2:env.env.dim//2+env.env.dim]):
                        correct_wm_cnt += 1
                        correct_wm = True
                if np.array_equal(nb.unit_dic['MinWMAgent1.ActionDeterminer'].inputs["observation"][-4:-2], [1, 1]):
                    if max(nb.unit_dic['MinWMAgent1.ActionDeterminer'].get_out_port("action").buffer) > 0:
                        ad_target_gone += 1
                        if np.array_equal(nb.unit_dic['MinWMAgent1.ActionDeterminer'].inputs["observation"][-2:],
                                          nb.unit_dic['MinWMAgent1.ActionDeterminer'].get_out_port("action").buffer):
                            ad_correct_cnt += 1
            if agent.env.done:
                break
        agent.env.flush = True
        while model.get_out_port("done").buffer[0] != 1:
            scheduler.step()
        if dump is not None and "a" in train["dump_flags"]:
            dump.write("reward: {0}\n".format(agent.get_in_port("reward").buffer[0]))
        if dump is not None and "s" in train["dump_flags"]:
            reward_sum += agent.get_in_port("reward").buffer[0]
            if nb.unit_dic['MinWMAgent1.ActionDeterminer'].gone:
                ad_gone_counter += 1
            if nb.unit_dic['MinWMAgent1.RegisterController'].gone:
                rc_gone_counter += 1
            if dump_counter % dump_cycle == 0 and dump_counter != 0:
                reward_sum_per_gone = reward_sum / ad_gone_counter if ad_gone_counter > 0 else 0.0
                rc_sample_per_gone = rc_sample_gone / rc_gone_counter if rc_gone_counter > 0 else 0.0
                correct_wm_per_gone = correct_wm_cnt / rc_gone_counter if rc_gone_counter > 0 else 0.0
                ad_target_per_gone = ad_target_gone / ad_gone_counter if ad_gone_counter > 0 else 0.0
                ad_correct = ad_correct_cnt / ad_gone_counter if ad_gone_counter > 0 else 0.0
                dump.write("{0}: avr. reward: {1:.2f}, reward per gone: {2:.2f}, ".
                           format(dump_counter // dump_cycle,
                                  reward_sum / dump_cycle,
                                  reward_sum_per_gone))
                dump.write("rc sg: {0:.2f}, rc cwm: {1:.2f}, ad tg: {2:.2f}, ad cr: {3:.2f}".
                    format(rc_sample_per_gone, correct_wm_per_gone, ad_target_per_gone, ad_correct))
                dump.write("\n")
                reward_sum = 0.0
                ad_gone_counter = 0
                rc_gone_counter = 0
                rc_sample_gone = 0
                correct_wm_cnt = 0
                ad_target_gone = 0
                ad_correct_cnt = 0
            dump_counter += 1
        nb.unit_dic['MinWMAgent1.ActionDeterminer'].reset()
        nb.unit_dic['MinWMAgent1.Register'].reset()
        nb.unit_dic['MinWMAgent1.RegisterController'].reset()
        agent.env.reset()
        # agent.env.out_ports['token_out'] = np.array([0])
        agent.env.done = False
    print("Close")
    nb.unit_dic['MinWMAgent1.ActionDeterminer'].close()
    nb.unit_dic['MinWMAgent1.RegisterController'].close()
    env.close()


if __name__ == '__main__':
    main()