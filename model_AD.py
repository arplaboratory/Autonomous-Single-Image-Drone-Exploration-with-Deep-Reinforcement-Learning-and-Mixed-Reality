import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_next_ad(input_data):
	yaw_new = (input_data['yaw'] + input_data['action'][:, :, 0].unsqueeze(-1) * 8)
	x_new = input_data['x'] + input_data['action'][:, :, 1].unsqueeze(-1) * 8 * torch.cos(yaw_new * np.pi / 180)
	y_new = input_data['y'] + input_data['action'][:, :, 1].unsqueeze(-1) * 8 * torch.sin(yaw_new * np.pi / 180)

	yaw_to_target = input_data['yaw'] - input_data['angle']
	yaw_to_target = torch.remainder(yaw_to_target + 180, 360) - 180

	target_x = input_data['x'] + input_data['distance'][:, :, 0].unsqueeze(-1) * torch.cos(yaw_to_target * np.pi / 180)
	target_y = input_data['y'] + input_data['distance'][:, :, 0].unsqueeze(-1) * torch.sin(yaw_to_target * np.pi / 180)

	new_distance_x = target_x - x_new
	new_distance_y = target_y - y_new
	d_next = (new_distance_x ** 2 + new_distance_y ** 2) ** (1 / 2)
	new_distance_x_norm = new_distance_x / d_next
	new_distance_y_norm = new_distance_y / d_next

	forward_x = torch.cos(yaw_new * np.pi / 180)
	forward_y = torch.sin(yaw_new * np.pi / 180)

	a_next = (torch.atan2(new_distance_x_norm, new_distance_y_norm) - torch.atan2(forward_x, forward_y)) / np.pi * 180
	a_next = torch.remainder(a_next + 180, 360) - 180

	return a_next, d_next


class CNN_feature_extr(nn.Module):
	def __init__(self, settings):
		super(CNN_feature_extr, self).__init__()
		self.channels = settings['channels']

		self.conv1 = nn.Conv2d(in_channels=settings['channels'], out_channels=settings['d1'], kernel_size=8, stride=4, padding=0)
		self.bnc1 = torch.nn.GroupNorm(int(settings['d1'] / 2), settings['d1'])

		self.conv2 = nn.Conv2d(in_channels=settings['d1'], out_channels=settings['d2'], kernel_size=4, stride=2, padding=0)
		self.bnc2 = torch.nn.GroupNorm(int(settings['d2'] / 2), settings['d2'])

	def forward(self, input_data):
		x = input_data['RGBCamera'].view(-1, 3, 84, 84)
		x1 = self.bnc1(F.relu(self.conv1(x)))
		x2 = self.bnc2(F.relu(self.conv2(x1)))
		return x2


class CNN_ADVAT(nn.Module):
	def __init__(self, settings):
		channels = settings['channels']
		d1 = settings['d1']
		d2 = settings['d2']
		super(CNN_ADVAT, self).__init__()
		self.conv1 = nn.Conv2d(channels, d1, 5, stride=1, padding=2)
		self.maxp1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(d1, d1, 5, stride=1, padding=1)
		self.maxp2 = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(d1, d2, 4, stride=1, padding=1)
		self.maxp3 = nn.MaxPool2d(2, 2)
		self.conv4 = nn.Conv2d(d2, d2, 3, stride=1, padding=1)
		self.maxp4 = nn.MaxPool2d(2, 2)

		relu_gain = nn.init.calculate_gain('relu')
		self.conv1.weight.data.mul_(relu_gain)
		self.conv2.weight.data.mul_(relu_gain)
		self.conv3.weight.data.mul_(relu_gain)
		self.conv4.weight.data.mul_(relu_gain)

	def forward(self, input_data):
		x = input_data['RGBCamera'].view(-1, 3, 84, 84)
		x1 = F.relu(self.maxp1(self.conv1(x)))
		x2 = F.relu(self.maxp2(self.conv2(x1)))
		x3 = F.relu(self.maxp3(self.conv3(x2)))
		x4 = F.relu(self.maxp4(self.conv4(x3)))
		return x4


class Actor_RGB(nn.Module):
	def __init__(self, settings):
		super(Actor_RGB, self).__init__()
		self.settings = settings
		self.state_space = settings['RGB_state_space']

		self.feature_extr = CNN_feature_extr(settings)

		width_1 = np.floor((self.state_space[0] - 8) / 4 + 1)
		width_2 = np.floor((width_1 - 4) / 2 + 1)

		height_1 = np.floor((self.state_space[1] - 8) / 4 + 1)
		height_2 = np.floor((height_1 - 4) / 2 + 1)

		self.linear_input = int(width_2 * height_2 * self.settings['d2'])

		self.lin = nn.Linear(self.linear_input, self.settings['dl'])
		self.gru = nn.GRU(self.settings['dl'], self.settings['dr'])

		self.hl1 = nn.Linear(self.settings['dr'], 200)
		self.hl2 = nn.Linear(200, 100)
		self.actor = nn.Linear(100, self.settings['output_a'])

	@property
	def learnable(self):
		return True

	def forward(self, input_data):
		shape = input_data['RGBCamera'].shape
		# Reshape per i layers convoluzionali
		x = self.feature_extr.forward(input_data)
		x_ = x.view(-1, self.linear_input)
		x1 = F.relu(self.lin(x_))

		# Reshape per la GRU
		x1_ = x1.view(shape[0], -1, self.settings['dl'])
		#hc = hc.view(1, -1, self.settings['dl'])

		x2, hn = self.gru(x1_)
		x2_ = F.relu(x2)

		# Actor
		x3 = F.relu(self.hl1(x2_))
		x4 = F.relu(self.hl2(x3))
		action = torch.tanh(self.actor(x4))  # Per limitare l'uscita tra -1 e 1
		# action = torch.clamp(self.actor(x4), -1, 1)

		return action, hn

	def init_hidden(self):
		h = np.zeros((1, 1, self.settings['dl']), dtype=np.float32)
		c = np.zeros((1, 1, self.settings['dl']), dtype=np.float32)
		return h, c

	def choose_action(self, input_data):
		self.eval()
		input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)
		action, hn = self.forward(input_data)

		output_data = {'action': action.detach().cpu().numpy().squeeze(), 'hc_': self.init_hidden()}
		return output_data

	def get_action(self, input_data):
		self.train()
		action, hn = self.forward(input_data)
		return action


class Actor_RGB_ADVAT(nn.Module):
	def __init__(self, settings):
		super(Actor_RGB_ADVAT, self).__init__()
		self.settings = settings
		self.state_space = settings['RGB_state_space']

		self.feature_extr = CNN_ADVAT(settings)

		width_1 = np.floor((self.state_space[0] - 5 + 4) / 1 + 1)
		width_1_max = np.floor((width_1 - 1 - 1) / 2 + 1)
		width_2 = np.floor((width_1_max - 5 + 2) / 1 + 1)
		width_2_max = np.floor((width_2 - 1 - 1) / 2 + 1)
		width_3 = np.floor((width_2_max - 4 + 2) / 1 + 1)
		width_3_max = np.floor((width_3 - 1 - 1) / 2 + 1)
		width_4 = np.floor((width_3_max - 3 + 2) / 1 + 1)
		width_4_max = np.floor((width_4 - 1 - 1) / 2 + 1)

		height_1 = np.floor((self.state_space[1] - 5 + 4) / 1 + 1)
		height_1_max = np.floor((height_1 - 1 - 1) / 2 + 1)
		height_2 = np.floor((height_1_max - 5 + 2) / 1 + 1)
		height_2_max = np.floor((height_2 - 1 - 1) / 2 + 1)
		height_3 = np.floor((height_2_max - 4 + 2) / 1 + 1)
		height_3_max = np.floor((height_3 - 1 - 1) / 2 + 1)
		height_4 = np.floor((height_3_max - 3 + 2) / 1 + 1)
		height_4_max = np.floor((height_4 - 1 - 1) / 2 + 1)

		self.linear_input = int(width_4_max * height_4_max * self.settings['d2'])

		self.lin = nn.Linear(self.linear_input, self.settings['dl'])
		self.gru = nn.GRU(self.settings['dl'], self.settings['dr'])

		self.hl1 = nn.Linear(self.settings['dr'], 200)
		self.hl2 = nn.Linear(200, 100)
		self.actor = nn.Linear(100, self.settings['output_a'])

	@property
	def learnable(self):
		return True

	def forward(self, input_data):
		shape = input_data['RGBCamera'].shape
		# Reshape per i layers convoluzionali
		x = self.feature_extr.forward(input_data)
		x_ = x.view(-1, self.linear_input)
		x1 = F.relu(self.lin(x_))

		# Reshape per la GRU
		x1_ = x1.view(shape[0], -1, self.settings['dl'])
		#hc = hc.view(1, -1, self.settings['dl'])

		x2, hn = self.gru(x1_)
		x2_ = F.relu(x2)

		# Actor
		x3 = F.relu(self.hl1(x2_))
		x4 = F.relu(self.hl2(x3))
		action = torch.tanh(self.actor(x4))  # Per limitare l'uscita tra -1 e 1
		# action = torch.clamp(self.actor(x4), -1, 1)

		return action, hn

	def init_hidden(self):
		h = np.zeros((1, 1, self.settings['dl']), dtype=np.float32)
		c = np.zeros((1, 1, self.settings['dl']), dtype=np.float32)
		return h, c

	def choose_action(self, input_data):
		self.eval()
		input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)
		action, hn = self.forward(input_data)

		output_data = {'action': action.detach().cpu().numpy().squeeze(), 'hc_': self.init_hidden()}
		return output_data

	def get_action(self, input_data):
		self.train()
		action, hn = self.forward(input_data)
		return action


class Actor(nn.Module):
	def __init__(self, settings):
		super(Actor, self).__init__()
		self.state_space = settings['state_space']
		self.settings = settings

		self.lin = nn.Linear(self.state_space, settings['dl'])
		self.gru = nn.GRU(settings['dl'], settings['dr'])

		self.hl1 = nn.Linear(settings['dr'], 200)
		self.hl2 = nn.Linear(200, 100)
		self.actor = nn.Linear(100, self.settings['output_a'])

	def forward(self, input_data):
		shape = input_data['angle'].shape
		# Reshape per i layers convoluzionali
		angle_distance = torch.cat((input_data['angle'] / 180, input_data['distance'] / 400), dim=-1)
		x1 = F.relu(self.lin(angle_distance))

		# Reshape per la GRU
		x1_ = x1.view(shape[0], -1, self.settings['dl'])
		hc = input_data['hc'].view(1, -1, self.settings['dl'])

		x2, hn = self.gru(x1_)
		x2_ = F.relu(x2)

		# Critic
		x3 = F.relu(self.hl1(x2_))
		x4 = F.relu(self.hl2(x3))
		action = torch.tanh(self.actor(x4))

		return action, hn


class Critic(nn.Module):
	def __init__(self, settings):
		super(Critic, self).__init__()
		self.state_space = settings['state_space']
		self.settings = settings

		self.lin = nn.Linear(self.state_space, settings['dl'])
		#self.gru = nn.GRU(settings['dl'], settings['dr'])

		self.hl1 = nn.Linear(settings['dr'], 200)
		self.hl2 = nn.Linear(200, 100)
		self.critic = nn.Linear(100, 1)

	def forward(self, input_data):
		shape = input_data['angle'].shape

		angle, distance = get_next_ad(input_data)
		angle_distance = torch.cat((angle / 180, distance / 400), dim=-1)
		x1 = F.relu(self.lin(angle_distance))

		# Reshape per la GRU
		#x1_ = x1.view(shape[0], -1, self.settings['dl'])
		#hc = input_data['hc'].view(1, -1, self.settings['dl'])

		#x2, hn = self.gru(x1_)
		#x2_ = F.relu(x2)

		# Critic
		q_value = F.relu(self.hl1(x1))
		q_value = F.relu(self.hl2(q_value))
		q_value = self.critic(q_value)

		return q_value, None


class DDPG_Net(nn.Module):
	def __init__(self, settings):
		super(DDPG_Net, self).__init__()
		self.settings = settings

		self.actor = Actor(settings)
		self.critic = Critic(settings)

	@property
	def learnable(self):
		return True

	def forward(self, input_data):
		if input_data.get('action', None) is None:
			out, hc = self.actor(input_data)
		else:
			out, hc = self.critic(input_data)
		return out

	@property
	def device(self):
		return next(self.parameters()).device

	def init_hidden(self):
		h = np.zeros((1, 1, self.settings['dl']), dtype=np.float32)
		c = np.zeros((1, 1, self.settings['dl']), dtype=np.float32)
		return h, c

	def choose_action(self, input_data):
		self.eval()
		input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)
		action = self.forward(input_data)

		output_data = {'action': action.detach().cpu().numpy().squeeze(), 'hc_': self.init_hidden()}
		return output_data

	def get_critic_test(self, input_data):
		self.eval()
		input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)

		q_value = self.forward(input_data)[0]
		if self.settings['TD3']:
			q_value = q_value[0]

		return q_value.detach().squeeze().cpu().numpy()

	def get_action(self, input_data):
		action = self.forward(input_data)
		return action

	def get_critic(self, input_data):
		self.train()
		q_value = self.forward(input_data)
		return q_value

	def get_action_critic_target(self, input_data):
		self.eval()
		action = self.forward(input_data)
		# if self.settings['TD3']:
		# 	action = action + torch.clamp(0.5 * torch.randn(action.shape), -0.5, 0.5).to(action.device)
		# 	action = torch.clamp(action, -1, 1)
		input_data['action'] = action

		q_value = self.forward(input_data)
		if self.settings['TD3']:
			q_value = torch.min(q_value[0], q_value[1])

		return q_value.data
