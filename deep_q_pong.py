"""
The greedy-epsilon hyperparameter will be decayed over the first 100k frames and then kept stable.
The buffer will hold 10k transitions.
"""
from lib import wrappers
from lib import q_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

# set environment and reward boundary for the last 100 episodes
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99 # used for Bellman approximation
BATCH_SIZE = 32 # batch size samples from the replay buffer
REPLAY_SIZE = 10000 # maximum capacity of the buffer
LEARNING_RATE = 1e-4 # learning rate for Adam optimizer
SYNC_TARGET_FRAMES = 1000 # how frequently to sync model weights from the training model to target model, which is used to get the value of tje next state in the Bellman approximation
REPLAY_START_SIZE = 10000 # frames to wait before starting training / populating replay buffer


EPSILON_DECAY_LAST_FRAME = 10**5 # linearly decayed over 100k frames
EPSILON_START = 1.0 # initialized to all random actions
EPSILON_FINAL = 0.02 # final epsilon in which only 2% of actions are random

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
class ExperienceBuffer:
    """
    Experience Replay buffer keeps the last transitions obtained from the environment, 
    a tuple of (observation, action, done flag, next state)
    Each step in the environment pushes the latest values into the buffer, constrained 
    to REPLAY_START_SIZE.
    When training, randomly sample the batch of transitions from the replay buffer, which
    breaks correlation between subsequent steps in the environment. 
    """
    def __init__(self, capacity):
        """
        Deque allows for fast appends and pops to maintain the number of buffer entries.
        """
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample from the buffer with random indices. 
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    """
    Interacts with the environment and saves the result of the interaction into the replay buffer.
    """
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        """
        The main method of the agent performs a step in the environment and stores 
        its result in the buffer. 
        Uses epsilon and either takes a random action, or uses the past model to obtain the 
        Q-values for all possible actions and chooses the best. 
        """
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment, get observation and reward
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp) # store data in experience buffer
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    """
    Arguments:
    batch: tuple created by sample() method from experience buffer
    net: first model which is used to calculate gradients
    tgt_net: second model used to calculate values for the next state, doesn't affect gradeitnss
    ~~~

    Calculate loss for the sampled batch. Instead of using a loop, use vectorized implementation.
    L =(Q(s,a) - y)^2 for steps that aren't at the end of the episode
    L = (Q(s,a) - r)^2 for the steps at the end
    """
    states, actions, rewards, dones, next_states = batch 

    # use GPU if specified in arguments
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    # comments refer the line above
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # pass observations to the first model and extract Q-values for the actions with gather()
    # 1 refers to index 1: actions, unsqueeze formats data for gather, squeeze gets rid of the extra dimensions created
    next_state_values = tgt_net(next_states_v).max(1)[0]
    # apply target network to next state observations and calculate maximum Q-value along dimension 1 (actions)
    # get index 0 of the max which is the values and not the indices
    next_state_values[done_mask] = 0.0
    # if the transition in the batch is from the last batch of the episode
    # the value of the action doesn't have a discounted reward of the state as there is no next state to gather reward from
    # necessary for convergence
    next_state_values = next_state_values.detach() 
    # detach() prevents gradients from flowing into the target network (the network used to calculate Q approximation for next states)
        # detach returns the tensor without connection to its calculation history
    # without doing this, the backprop of the loss will start to affect predictions for both the current and next state
    # but the next state's predictions need to be kept clean because they are used in the Bellman equation to calculate reference Q-values

    expected_state_action_values = next_state_values * GAMMA + rewards_v # calculate Bellman approximation value
    return nn.MSELoss()(state_action_values, expected_state_action_values) # calculate mse loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stopping training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # create environnment with all wrappers, define neural network and target networks
    # the two networks are initialized with different random weights but it doesn't matter as they will be synced
    #   every 1000 frames
    env = wrappers.make_env(args.env)
    net = q_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = q_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)  # create experience buffer
    agent = Agent(env, buffer) # pass env and buffer to agent
    epsilon = EPSILON_START # initialize epsilon with starting value

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = [] # buffer for full rewards
    frame_idx = 0 # counter for current frame
    ts_frame = 0 
    ts = time.time()
    best_mean_reward = None 

    # TRAINING LOOP
    while True:
        frame_idx += 1 # count iterations completed
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME) # defining epsilon decay
        
        reward = agent.play_step(net, epsilon, device=device) # make a single step in the environment using current network and epsilon value
        # returns a non-None result only if this is the final step in the episode
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts) # speed as a count of frames processed per second
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                # every time the mean reward beats the best, report and save the model parameters
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break
        # if mean reward beats the record, save the model
        # boundary set to 19.5 so model has to beat 19/21 games

        if len(buffer) < REPLAY_START_SIZE: # check if buffer is large enough for training
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0: # sync params from main to target
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad() # zero gradients
        batch = buffer.sample(BATCH_SIZE) # sample data batches
        loss_t = calc_loss(batch, net, tgt_net, device=device) # calculate loss
        loss_t.backward()
        optimizer.step() # minimize loss
    writer.close()
