import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class FeaturesExtractor(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor, self).__init__()
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.d1,
                               kernel_size=3, stride=1, padding=0)
        self.maxp1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=0)
        self.maxp2 = nn.MaxPool2d(2, 2)

    def forward(self, input_data):
        x = input_data['RGBCamera'].reshape(-1, 3, 84, 84)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        return x


class Actor(nn.Module):
    def __init__(self, settings):
        super(Actor, self).__init__()
        self.settings = settings
        self.state_space = settings['RGB_state_space']
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.dr = settings['dr']
        self.dl = settings['dl']

        # features extraction
        self.features = FeaturesExtractor(settings)

        # shape out convolution
        self.pixel = np.floor((((self.state_space[0] - 2)/2) - 2)/2)  # 19
        self.in_shape = 19*19*64  # for square images

        # recurrent unit
        self.lin = nn.Linear(self.in_shape, self.dr)
        self.gru = nn.GRU(self.dr, self.dl)

        # hidden layer
        self.hl = nn.Linear(self.dl, 100)

        # output
        self.action = nn.Linear(100, self.settings['output_a'])

    def forward(self, input_data):
        shape = input_data['RGBCamera'].shape[0]

        # reshape
        x = self.features.forward(input_data)
        x = x.view(-1, self.in_shape)
        xr = F.relu(self.lin(x))

        # GRU
        xg = xr.view(shape, -1, self.dr)
        xg, h_n = self.gru(xg)

        # action
        xa = F.relu(self.hl(xg))
        action = torch.tanh(self.action(xa))
        return action, h_n


class Critic(nn.Module):
    def __init__(self, settings):
        super(Critic, self).__init__()
        self.settings = settings
        self.state_space = settings['RGB_state_space']
        self.channels = settings['channels']
        self.dr = settings['dr']
        self.dl = settings['dl']

        # features extraction
        self.features = FeaturesExtractor(settings)

        # shape out convolution
        self.out_shape = 19*19*64  # for square images
        self.cat_shape = (19*19*64)+2

        # fully connected
        self.lin = nn.Linear(self.cat_shape, self.dr)

        # recurrent unit
        self.gru = nn.GRU(self.dr, self.dl)

        # hidden
        self.hl1 = nn.Linear(self.dl, 200)
        self.hl2 = nn.Linear(200, 100)

        # output
        self.critic = nn.Linear(100, 1)

    def forward(self, input_data):
        shape = input_data['RGBCamera'].shape[0]
        # reshape
        x = self.features.forward(input_data)
        x = x.view(-1, self.out_shape)

        # concat action
        xc = torch.cat((x, input_data['action'].view(-1, 2)), dim=1)

        # network
        xn = F.relu(self.lin(xc))
        xg = xn.view(shape, -1, self.dr)
        xg, _ = self.gru(xg)
        xg = F.relu(self.hl1(xg))
        xh = F.relu(self.hl2(xg))

        # critic
        q_value = self.critic(xh)
        return q_value, None


class DDPG_Net(nn.Module):
    def __init__(self, settings):
        super(DDPG_Net, self).__init__()
        self.settings = settings

        self.actor = Actor(settings)
        self.critic = Critic(settings)
        self.critic_1 = Critic(settings)

    @property
    def learnable(self):
        return True

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_data):
        if input_data.get('action', None) is None:
            out, hc = self.actor(input_data)
        else:
            if self.settings['TD3']:
                out0, hc = self.critic(input_data)
                out1, hc = self.critic_1(input_data)
                out = torch.zeros((2, self.settings['seq_len'], self.settings['batch_size'], 1)).to('cuda:0')
                out[0] = out0
                out[1] = out1
            else:
                out, hc = self.critic(input_data)
        return out

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
        input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(
            0)
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
        if self.settings['TD3']:
            action = action + torch.clamp(0.5 * torch.randn(action.shape), -0.5, 0.5).to(action.device)
            action = torch.clamp(action, -1, 1)
        input_data['action'] = action
        q_value = self.forward(input_data)
        if self.settings['TD3']:
            q_value = torch.min(q_value[0], q_value[1])
        return q_value.data


########################################################################################################################


class FeaturesExtractorIMPALA(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractorIMPALA, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']

        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.d1,
                               kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=4, stride=2, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input_data):
        inp = input_data['RGBCamera'].reshape(-1, 3, 84, 84)
        x1 = F.relu(self.conv1(inp))
        x2 = F.relu(self.maxp(self.conv2(x1)))
        return x2


class ActorIMPALA(nn.Module):
    def __init__(self, settings):
        super(ActorIMPALA, self).__init__()
        self.settings = settings
        self.state_space = settings['RGB_state_space']
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.dl = settings['dl']
        self.dr = settings['dr']

        self.features = FeaturesExtractorIMPALA(settings)
        width_conv1 = int((self.state_space[0]-8)/4 + 1)
        height_conv1 = int((self.state_space[0]-8)/4 + 1)
        width_conv2 = int((width_conv1-4+2*1)/2 + 1)
        height_conv2 = int((height_conv1-4+2*1)/2 + 1)
        width_maxp = int((width_conv2-2)/2 + 1)
        height_maxp = int((height_conv2-2)/2 + 1)
        self.feat_shape = width_maxp * height_maxp * self.d2  # 5*5*64=1600

        self.lin = nn.Linear(self.feat_shape, self.dl)
        self.gru = nn.GRU(self.dl, self.dr)
        self.hidden1 = nn.Linear(self.dr, 128)
        self.hidden2 = nn.Linear(128, 64)

        self.output = nn.Linear(64, self.settings['output_a'])

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        # print('shape', input_data['RGBCamera'].shape)

        features = self.features.forward(input_data)
        features_ = features.view(-1, self.feat_shape)

        x = F.relu(self.lin(features_))
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, hn = self.gru(x_)

        xh1 = F.relu(self.hidden1(xgru))
        xh2 = F.relu(self.hidden2(xh1))

        logits = self.output(xh2)  # TODO: check shape

        return logits, hn


class CriticIMPALA(nn.Module):
    def __init__(self, settings):
        super(CriticIMPALA, self).__init__()
        self.settings = settings
        self.state_space = settings['RGB_state_space']
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.dl = settings['dl']
        self.dr = settings['dr']

        self.features = FeaturesExtractorIMPALA(settings)
        width_conv1 = int((self.state_space[0] - 8) / 4 + 1)
        height_conv1 = int((self.state_space[0] - 8) / 4 + 1)
        width_conv2 = int((width_conv1 - 4 + 2 * 1) / 2 + 1)
        height_conv2 = int((height_conv1 - 4 + 2 * 1) / 2 + 1)
        width_maxp = int((width_conv2 - 2) / 2 + 1)
        height_maxp = int((height_conv2 - 2) / 2 + 1)
        self.feat_shape = width_maxp * height_maxp * self.d2  # 5*5*64=1600

        self.lin = nn.Linear(self.feat_shape, self.dl)
        self.gru = nn.GRU(self.dl, self.dr)
        self.hidden1 = nn.Linear(self.dr, 128)
        self.hidden2 = nn.Linear(128, 64)

        self.output = nn.Linear(64, 1)

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape

        features = self.features.forward(input_data)
        features_ = features.view(-1, self.feat_shape)

        x = F.relu(self.lin(features_))
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, hn = self.gru(x_)

        xh1 = F.relu(self.hidden1(xgru))
        xh2 = F.relu(self.hidden2(xh1))

        values = self.output(xh2)  # TODO: check shape

        return values, hn


class IMPALA_Net(nn.Module):
    def __init__(self, settings):
        super(IMPALA_Net, self).__init__()
        self.settings = settings
        self.actor = ActorIMPALA(settings)
        self.critic = CriticIMPALA(settings)

    @property
    def learnable(self):
        return True

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_data):
        logits, hc = self.actor(input_data)
        values, hc = self.critic(input_data)
        return logits, values, hc

    def choose_action(self, input_data):
        self.eval()
        input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)
        logits, _, hc = self.forward(input_data)
        probs = torch.clamp(F.softmax(logits, dim=-1), 0.00001, 0.99999).data  # .values()
        m = torch.distributions.Categorical(probs)
        action = m.sample().type(torch.IntTensor)
        output_data = {'action': action.detach().cpu().numpy().squeeze(), 'logits': logits, 'hc_': self.init_hidden()}
        return output_data

    def get_logits(self, input_data):
        self.train()
        logits, _, _ = self.forward(input_data)
        return logits

    def get_values(self, input_data):
        self.eval()
        _, values, _ = self.forward(input_data)
        return values

    def init_hidden(self):
        h = np.zeros((1, 1, self.settings['dl']), dtype=np.float32)
        c = np.zeros((1, 1, self.settings['dl']), dtype=np.float32)
        return h, c

