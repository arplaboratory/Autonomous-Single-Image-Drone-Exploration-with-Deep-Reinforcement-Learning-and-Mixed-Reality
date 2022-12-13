import torch
import torch.nn.functional as F
import torch.nn as nn


###################################
class FeaturesExtractor(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.d1,
                               kernel_size=5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=0)
        self.maxp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f_shape = 32 * 6 * 6

    def forward(self, input_data):
        inp = input_data['RGBCamera'].reshape(-1, 3, 128, 128)

        x = F.relu(self.maxp1(self.conv1(inp)))
        x1 = F.relu(self.maxp2(self.conv2(x)))
        x2 = F.relu(self.maxp3(self.conv3(x1)))
        x3 = F.relu(self.maxp4(self.conv4(x2)))

        return x3
###################################


###################################
class FeaturesExtractor1(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor1, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.d1,
                               kernel_size=8, stride=4, padding=2)
        self.bnc1 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d1,
                               kernel_size=4, stride=2, padding=1)
        self.bnc2 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv3 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=1)
        self.bnc3 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)
        self.conv4 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=0)
        self.bnc4 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)

        self.f_shape = 32 * 6 * 6

    def forward(self, input_data):
        inp = input_data['RGBCamera'].reshape(-1, 3, 128, 128)

        x1 = self.maxp(inp)
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5
###################################


###################################
class FeaturesExtractor2(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor2, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']
        self.img = torch.zeros(3, 84, 84)

        self.lnorm1 = nn.LayerNorm(self.img.shape)
        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.d1,
                               kernel_size=4, stride=2, padding=1)
        self.lnorm2 = nn.LayerNorm([self.d1, 42, 42])
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=4, stride=2, padding=1)
        self.lnorm3 = nn.LayerNorm([self.d2, 21, 21])
        self.conv3 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=1)
        self.lnorm4 = nn.LayerNorm([self.d2, 10, 10])
        self.conv4 = nn.Conv2d(in_channels=self.d2, out_channels=self.d3,
                               kernel_size=3, stride=1, padding=1)
        self.lnorm5 = nn.LayerNorm([self.d3, 10, 10])
        self.conv5 = nn.Conv2d(in_channels=self.d3, out_channels=self.d3,
                               kernel_size=3, stride=1, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.f_shape = 64*5*5

    def forward(self, input_data):
        inp = input_data['RGBCamera'].reshape(-1, 3, 84, 84)
        x1 = F.relu(self.conv1(self.lnorm1(inp)))
        x2 = F.relu(self.conv2(self.lnorm2(x1)))
        x3 = F.relu(self.maxp(self.conv3(self.lnorm3(x2))))
        x4 = F.relu(self.conv4(self.lnorm4(x3)))
        x5 = F.relu(self.maxp(self.conv5(self.lnorm5(x4))))

        return x5
###################################


class ActorIMPALA(nn.Module):
    def __init__(self, settings):
        super(ActorIMPALA, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.dl = settings['dl']
        self.dr = settings['dr']

        self.features = FeaturesExtractor1(settings)
        self.feat_shape = self.features.f_shape

        self.lin = nn.Linear(self.feat_shape, self.dl)
        self.gru = nn.GRU(self.dl, self.dr)
        self.output = nn.Linear(self.dr, self.settings['output_a'])

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_ha = input_data['h_a']

        features = self.features.forward(input_data)
        features_ = features.view(-1, self.feat_shape)

        x = F.relu(self.lin(features_))
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, h_a_ = self.gru(x_, input_ha)

        logits = self.output(xgru)

        return logits, h_a_


class CriticIMPALA(nn.Module):
    def __init__(self, settings):
        super(CriticIMPALA, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.dl = settings['dl']
        self.dr = settings['dr']

        self.features = FeaturesExtractor1(settings)
        self.feat_shape = self.features.f_shape

        self.lin = nn.Linear(self.feat_shape, self.dl)
        self.gru = nn.GRU(self.dl, self.dr)
        self.output = nn.Linear(self.dr, 1)

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_hc = input_data['h_c']

        features = self.features.forward(input_data)
        features_ = features.view(-1, self.feat_shape)

        x = F.relu(self.lin(features_))
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, h_c_ = self.gru(x_, input_hc)

        values = self.output(xgru)

        return values, h_c_


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
        logits, h_a_ = self.actor(input_data)
        values, h_c_ = self.critic(input_data)

        return logits, values, h_a_, h_c_

    def choose_action(self, input_data):
        self.eval()
        input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)
        input_data['h_a'] = torch.from_numpy(input_data['h_a']).float().to(input_data['device'])
        input_data['h_c'] = torch.from_numpy(input_data['h_c']).float().to(input_data['device'])
        logits, _, h_a_, h_c_ = self.forward(input_data)

        probs = torch.clamp(F.softmax(logits, dim=-1), 0.00001, 0.99999).data  # .values()
        m = torch.distributions.Categorical(probs)
        action = m.sample().type(torch.IntTensor)
        output_data = {'action': action.detach().cpu().numpy().squeeze(), 'logits': logits, 'h_a_': h_a_,  'h_c_': h_c_}

        return output_data

    def get_logits(self, input_data):
        self.train()
        logits, _, _, _ = self.forward(input_data)

        return logits

    def get_values(self, input_data):
        self.eval()
        _, values, _, _ = self.forward(input_data)

        return values

    def init_hidden(self):
        h = torch.zeros((1, 1, self.settings['dr']), dtype=torch.float32)
        return h

