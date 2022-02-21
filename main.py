from multiprocessing.spawn import import_main_path
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from env import TradeEnv
from model import ValueNetwork, PolicyNetwork
from misc import AverageMeter, ReplayBuffer
import matplotlib.pyplot as plt
import random
from tensorboardX import SummaryWriter
import os
import logging
from easydict import EasyDict


# basic settings
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
max_frames  = 300000
max_steps   = 300000
frame_idx   = 0
rewards     = []
batch_size  = 128

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

save_path = 'checkpoints.pth.tar'
event_path = 'events'
result_path = 'results'
log_path = 'log.txt'
makedir(event_path)
makedir(result_path)


# loggers
tb_logger = SummaryWriter(event_path)
logger = logging.getLogger()
formatter = logging.Formatter(
    '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
fh = logging.FileHandler(log_path)
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(sh)


# load states
try:
    state = torch.load(save_path, 'cpu')
    logger.info(f"Recovering from {save_path}")
except:
    state = {}
    state['last_frame'] = 0

frame_idx = state['last_frame']


# start env
logger.info("trying start env")
env = TradeEnv("GOLD.csv", "bitcoin.csv", "bitcoin_pre.txt","gold_pre.txt")


state_dim  = len(env.states)
action_dim = len(env.actions)
hidden_dim = 256
logger.info(f"the net setting is \n state dim: {state_dim} \n action dim {action_dim} \n hidden_dim{hidden_dim}")

# load models
value_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

target_value_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
logger.info("model load successufully")


if 'value_net' in state:
    value_net.load_state_dict(state['value_net'], strict=False)
if 'policy_net' in state:
    policy_net.load_state_dict(state['policy_net'], strict=False)
if 'target_value_net' in state:
    target_value_net.load_state_dict(state['target_value_net'], strict=False)
if 'target_policy_net' in state:
    target_policy_net.load_state_dict(state['target_policy_net'], strict=False)



for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)
    

# trainning settings
value_lr  = 3e-3
policy_lr = 1e-3

value_optimizer  = optim.Adam(value_net.parameters(),  lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

value_criterion = nn.MSELoss()

replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)
saved_freq = 100


# tb_logger settings
meters = EasyDict()
meters.p_losses = AverageMeter(saved_freq)
meters.v_losses = AverageMeter(saved_freq)
meters.reward = AverageMeter(saved_freq)
meters.money = AverageMeter(saved_freq)
meters.eval = AverageMeter(0)
step_count = 0

def ddpg_update(batch_size, 
           gamma = 0.99,
           min_value=-100,
           max_value=100,
           soft_tau=1e-2):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    meters.reward.update(torch.FloatTensor([np.mean(reward)]))
    meters.money.update(torch.FloatTensor([np.mean(next_state, axis=0)[-1]]))
    policy_net.train()
    target_policy_net.train()
    value_net.train()
    target_value_net.train()

    global step_count
    step_count += 1
    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    
    action_tmp = policy_net(state)
    policy_loss_co = 0
    for a, s  in zip(action_tmp, state):
        policy_loss_co += max(- a[0] - s[3], 0) + max(-a[1] - s[2], 0) + max((a[0] + a[1]) * 1000 - s[1], 0)
    policy_loss_co /= state.shape[0]

    policy_loss = value_net(state, policy_net(state))
    policy_loss = - policy_loss.mean() * 0.5 + policy_loss_co

    next_action    = target_policy_net(next_state)
    target_value   = target_value_net(next_state, next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())


    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    meters.p_losses.update(policy_loss)
    meters.v_losses.update(value_loss)
    if step_count % saved_freq == 0:
        tb_logger.add_scalar('p_loss_train', meters.p_losses.avg, step_count)
        tb_logger.add_scalar('v_loss_train', meters.v_losses.avg, step_count)
        tb_logger.add_scalar('reward_train', meters.reward.avg, step_count)
        tb_logger.add_scalar('money_train', meters.money.avg, step_count)

        log_msg = f'Iter: [{step_count}]\t' \
                f'p_Loss {policy_loss}:{meters.p_losses.val:.4f} ({meters.p_losses.avg:.4f})\t' \
                f'v_Loss {value_loss}:{meters.v_losses.val:.3f} ({meters.v_losses.avg:.3f})\t' \
                f'reward {meters.reward.val:.3f} ({meters.reward.avg:.3f})\t' \
                f'the money is {meters.money.val:.3f} ({meters.money.avg:.3f})'
        logger.info(log_msg)
 
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


def eval():
    logger.info(f"finish a round and eval the model....")
    state = env.reset()
    while True:
        action = policy_net.get_action(state)
        state, reward, done = env.step(state, action)
        if done:
            meters.eval.update(torch.FloatTensor([state[-1]]))
            tb_logger.add_scalar('eval', state[-1], step_count)
            logger.info(f'eval_final_money {state[-1]} ({meters.eval.avg:.3f})')
            return


while frame_idx < max_frames:
    logger.debug(f'start frame: {frame_idx}')
    state = env.reset()
    # ou_noise.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        action = policy_net.get_action(state)

        # action = ou_noise.get_action(action, step)
        next_state, reward, done = env.step(state, action)
        replay_buffer.push(state, action, reward, next_state, done)
        if len(replay_buffer) > batch_size:
            ddpg_update(batch_size)

        episode_reward += reward
        frame_idx += 1
        if done:
            eval()
            break
        state = next_state

    rewards.append(episode_reward)
    logger.debug(f"frame{frame_idx}: episode_reward is {episode_reward}")
    state_dict = {}
    
    state_dict['value_net'] = value_net.state_dict()
    state_dict['policy_net'] = policy_net.state_dict()
    state_dict['target_value_net'] = target_value_net.state_dict()
    state_dict['target_policy_net'] = target_policy_net.state_dict()
    state_dict['last_frame'] = frame_idx
    torch.save(state_dict, save_path)