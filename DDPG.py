from utils import get_input_data_DDPG
import torch
import copy


class DDPG:
	def __init__(self, settings):
		# Params
		self.settings = settings

		# GPU
		if torch.cuda.is_available() and self.settings['learner_gpu_id'] is not None:
			self.device = torch.device('cuda:' + str(self.settings['learner_gpu_id']))
		else:
			self.device = torch.device('cpu')

		# Networks
		self.actor_critic = self.settings['net']['net'].to(self.device)
		self.actor_critic_target = copy.deepcopy(self.settings['net']['net']).to(self.device)

		self.actor_critic_target.eval()

		actor_optimizer = torch.optim.Adam(self.actor_critic.actor.parameters(), lr=self.settings['lr_actor'])
		critic_optimizer = torch.optim.Adam(list(self.actor_critic.critic.parameters()) + list(self.actor_critic.critic_1.parameters()) if self.settings['TD3'] else self.actor_critic.critic.parameters(), lr=self.settings['lr_critic'])
		self.opt = [actor_optimizer, critic_optimizer]

		self.policy_update = True

	def compute_critic_loss_and_gradients(self, input_data):
		assert input_data['done'].shape == (self.settings['seq_len'], self.settings['batch_size'], 1)
		assert input_data['rewards'].shape == (self.settings['seq_len'], self.settings['batch_size'], 1)
		# Backpropagation
		Qvals = self.actor_critic.get_critic(input_data['get_critic'])
		next_Q = self.actor_critic_target.get_action_critic_target(input_data['get_action_critic_target'])

		assert next_Q.shape == (self.settings['seq_len'], self.settings['batch_size'], 1)
		assert not next_Q.requires_grad

		if self.settings['TD3']:
			buffer_v_target = []
			next_Q = next_Q[-1] * (- input_data['done'][-1] + 1)
			rewards = torch.flip(input_data['rewards'], [0])
			for r in rewards:
				next_Q = r + self.settings['gamma'] * next_Q
				buffer_v_target.append(next_Q)
			buffer_v_target.reverse()
			Qprime = torch.stack(buffer_v_target, dim=0)
		else:
			Qprime = input_data['rewards'] + next_Q * self.settings['gamma'] * (1 - input_data['done'])
		assert not Qprime.requires_grad

		if self.settings['TD3']:
			assert Qvals[0].shape == (self.settings['seq_len'], self.settings['batch_size'], 1)
			assert Qvals[1].shape == (self.settings['seq_len'], self.settings['batch_size'], 1)
			critic_loss = ((Qvals[0] - Qprime) ** 2 + (Qvals[1] - Qprime) ** 2).mean()
		else:
			assert Qvals.shape == (self.settings['seq_len'], self.settings['batch_size'], 1)
			critic_loss = ((Qvals - Qprime) ** 2).mean()

		self.actor_critic.zero_grad()
		critic_loss.backward(retain_graph=True)

		# Critic Grad
		grad_norm_critic = 0
		for name1, net1 in self.actor_critic.critic.named_parameters():
			if net1.grad is None:
				print("Gradients None Name: ", name1)
			else:
				grad_norm_critic += net1.grad.pow(2).sum()
		grad_norm_critic = grad_norm_critic ** (1 / 2)

		return critic_loss, grad_norm_critic

	def compute_actor_loss_and_gradients(self, input_data):
		actions_actor = self.actor_critic.get_action(input_data['get_action'])

		input_data['get_critic']['action'] = actions_actor
		q_value = self.actor_critic.get_critic(input_data['get_critic'])
		if self.settings['TD3']:
			q_value = q_value[0]
		assert q_value.shape == (self.settings['seq_len'], self.settings['batch_size'], 1)
		policy_loss = -q_value.mean()

		self.actor_critic.zero_grad()
		policy_loss.backward()

		# Policy Gradient
		grad_norm_policy = 0
		for name1, net1 in self.actor_critic.actor.named_parameters():
			if net1.grad is None:
				print("Gradienti None Name: ", name1)
			else:
				grad_norm_policy += net1.grad.pow(2).sum()
		grad_norm_policy = grad_norm_policy ** (1 / 2)

		return policy_loss, grad_norm_policy

	def on_policy_update(self, agent_state_dict, samples):
		shared_critic_state_dict = copy.deepcopy(self.actor_critic.critic.state_dict())
		shared_actor_state_dict = copy.deepcopy(self.actor_critic.actor.state_dict())
		self.actor_critic.load_state_dict(agent_state_dict)

		input_data = get_input_data_DDPG(samples, self.device)

		critic_loss, grad_norm_critic = self.compute_critic_loss_and_gradients(input_data)
		self.actor_critic.critic.load_state_dict(shared_critic_state_dict)
		self.opt[1].step()

		if self.policy_update:
			policy_loss, grad_norm_policy = self.compute_actor_loss_and_gradients(input_data)
			self.actor_critic.actor.load_state_dict(shared_actor_state_dict)
			self.opt[0].step()

			# Update target networks
			for target_param, param in zip(self.actor_critic_target.parameters(), self.actor_critic.parameters()):
				target_param.data.copy_(param.data * self.settings['tau'] + target_param.data * (1.0 - self.settings['tau']))

			if self.settings['TD3']:
				self.policy_update = not self.policy_update
		else:
			self.actor_critic.actor.load_state_dict(shared_actor_state_dict)
			policy_loss = torch.zeros((1))
			grad_norm_policy = torch.zeros((1))
			self.policy_update = not self.policy_update

		critic_loss = critic_loss.detach().cpu().numpy()
		policy_loss = policy_loss.detach().cpu().numpy()
		grad_norm_policy = grad_norm_policy.detach().cpu().numpy()
		grad_norm_critic = grad_norm_critic.detach().cpu().numpy()

		output_data = {
			'critic_loss': critic_loss,
			'policy_loss': policy_loss,
			'grad_norm_policy': grad_norm_policy,
			'grad_norm_critic': grad_norm_critic
		}

		return output_data

	def update(self, samples):

		input_data = get_input_data_DDPG(samples, self.device)

		critic_loss, grad_norm_critic = self.compute_critic_loss_and_gradients(input_data)
		self.opt[1].step()

		if self.policy_update:
			policy_loss, grad_norm_policy = self.compute_actor_loss_and_gradients(input_data)
			self.opt[0].step()

			# Update target networks
			for target_param, param in zip(self.actor_critic_target.parameters(), self.actor_critic.parameters()):
				target_param.data.copy_(param.data * self.settings['tau'] + target_param.data * (1.0 - self.settings['tau']))

			if self.settings['TD3']:
				self.policy_update = not self.policy_update
		else:
			policy_loss = torch.zeros((1))
			grad_norm_policy = torch.zeros((1))
			self.policy_update = not self.policy_update

		critic_loss = critic_loss.detach().cpu().numpy()
		policy_loss = policy_loss.detach().cpu().numpy()
		grad_norm_policy = grad_norm_policy.detach().cpu().numpy()
		grad_norm_critic = grad_norm_critic.detach().cpu().numpy()

		output_data = {
						'critic_loss': critic_loss,
						'policy_loss': policy_loss,
						'grad_norm_policy': grad_norm_policy,
						'grad_norm_critic': grad_norm_critic
					}

		return output_data
