import torch
import torch.nn.functional as F
import torch.nn as nn


class FeaturesExtractor_RGB(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_RGB, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.width = settings['width']
        self.height = settings['height']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

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

        self.feat_shape = 32 * 8 * 8

    def forward(self, input_data):
        inp = input_data['RGBCamera'].reshape(-1, self.channels, self.width, self.height)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5


class FeaturesExtractor_R(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_R, self).__init__()
        self.settings = settings
        self.grid = settings['grid_cells']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.d1,
                               kernel_size=4, stride=2, padding=2)
        self.bnc1 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d1,
                               kernel_size=3, stride=2, padding=1)
        self.bnc2 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv3 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=1)
        self.bnc3 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)
        self.conv4 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=0)
        self.bnc4 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)

        self.feat_shape = 32 * 4 * 4
        self.feat_shape = 32 * 7 * 7

    def forward(self, input_data):
        inp = input_data['obstacle_gt'].reshape(-1, 1, self.grid, self.grid)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5


class FeaturesExtractor_G(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_G, self).__init__()
        self.settings = settings
        self.grid = settings['grid_cells']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.d1,
                               kernel_size=4, stride=2, padding=2)
        self.bnc1 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d1,
                               kernel_size=3, stride=2, padding=1)
        self.bnc2 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv3 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=1)
        self.bnc3 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)
        self.conv4 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=0)
        self.bnc4 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)

        self.feat_shape = 32 * 4 * 4
        self.feat_shape = 32 * 7 * 7

    def forward(self, input_data):
        inp = input_data['tracker_estimate'].reshape(-1, 1, self.grid, self.grid)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5


class FeaturesExtractor_B(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_B, self).__init__()
        self.settings = settings
        self.grid = settings['grid_cells']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.d1,
                               kernel_size=4, stride=2, padding=2)
        self.bnc1 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d1,
                               kernel_size=3, stride=2, padding=1)
        self.bnc2 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv3 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=1)
        self.bnc3 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)
        self.conv4 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=0)
        self.bnc4 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)

        self.feat_shape = 32 * 4 * 4
        self.feat_shape = 32 * 7 * 7

    def forward(self, input_data):
        inp = input_data['target_estimate'].reshape(-1, 1, self.grid, self.grid)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5


class ActorIMPALA(nn.Module):
    def __init__(self, settings):
        super(ActorIMPALA, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']
        self.dl = settings['dl']
        self.dr = settings['dr']
        self.map_in = 100

        self.features_RGB = FeaturesExtractor_RGB(settings)
        self.features_R = FeaturesExtractor_R(settings)
        self.features_G = FeaturesExtractor_G(settings)
        self.features_B = FeaturesExtractor_B(settings)

        self.RGB_shape = self.features_RGB.feat_shape
        self.R_shape = self.features_R.feat_shape
        self.G_shape = self.features_G.feat_shape
        self.B_shape = self.features_B.feat_shape

        self.feat_shape = self.RGB_shape + self.R_shape + self.G_shape + self.B_shape

        # Linear
        self.lin1 = nn.Linear(self.feat_shape, 750)
        self.lin2 = nn.Linear(750, self.dl)

        # GRU
        self.gru = nn.GRU(self.dl, self.dr)

        # hidden
        self.hidden1 = nn.Linear(self.dr, 128)
        self.hidden2 = nn.Linear(128, 64)

        # output
        self.output = nn.Linear(64, self.settings['output_a'])

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_ha = input_data['h_a']

        featuresRGB = self.features_RGB.forward(input_data).reshape(-1, self.RGB_shape)
        featuresR = self.features_R.forward(input_data).reshape(-1, self.R_shape)
        featuresG = self.features_G.forward(input_data).reshape(-1, self.G_shape)
        featuresB = self.features_B.forward(input_data).reshape(-1, self.B_shape)

        features = torch.cat((featuresRGB, featuresR, featuresG, featuresB), dim=1)

        # lin
        x1 = F.relu(self.lin1(features))
        x2 = F.relu(self.lin2(x1))

        # gru
        x_ = x2.view(input_shape[0], -1, self.dl)
        xgru, h_a_ = self.gru(x_, input_ha)

        # hidden
        xh1 = F.relu(self.hidden1(xgru))
        xh2 = F.relu(self.hidden2(xh1))

        logits = self.output(xh2)

        return logits, h_a_


class CriticIMPALA(nn.Module):
    def __init__(self, settings):
        super(CriticIMPALA, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']
        self.dl = settings['dl']
        self.dr = settings['dr']
        self.map_in = 100

        self.features_RGB = FeaturesExtractor_RGB(settings)
        self.features_R = FeaturesExtractor_R(settings)
        self.features_G = FeaturesExtractor_G(settings)
        self.features_B = FeaturesExtractor_B(settings)

        self.RGB_shape = self.features_RGB.feat_shape
        self.R_shape = self.features_R.feat_shape
        self.G_shape = self.features_G.feat_shape
        self.B_shape = self.features_B.feat_shape

        self.feat_shape = self.RGB_shape + self.R_shape + self.G_shape + self.B_shape

        # Linear
        self.lin1 = nn.Linear(self.feat_shape, 750)
        self.lin2 = nn.Linear(750, self.dl)

        # GRU
        self.gru = nn.GRU(self.dl, self.dr)

        # hidden
        self.hidden1 = nn.Linear(self.dr, 128)
        self.hidden2 = nn.Linear(128, 64)

        # output
        self.output = nn.Linear(64, 1)

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_hc = input_data['h_c']

        featuresRGB = self.features_RGB.forward(input_data).reshape(-1, self.RGB_shape)
        featuresR = self.features_R.forward(input_data).reshape(-1, self.R_shape)
        featuresG = self.features_G.forward(input_data).reshape(-1, self.G_shape)
        featuresB = self.features_B.forward(input_data).reshape(-1, self.B_shape)

        features = torch.cat((featuresRGB, featuresR, featuresG, featuresB), dim=1)

        # lin
        x1 = F.relu(self.lin1(features))
        x2 = F.relu(self.lin2(x1))

        # gru
        x_ = x2.view(input_shape[0], -1, self.dl)
        xgru, h_c_ = self.gru(x_, input_hc)

        # hidden
        xh1 = F.relu(self.hidden1(xgru))
        xh2 = F.relu(self.hidden2(xh1))

        values = self.output(xh2)

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
        output_data = {'logits': logits, 'values': values, 'h_a_': h_a_, 'h_c_': h_c_}

        return output_data

    def choose_action(self, input_data):
        self.eval()
        input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)
        input_data['h_a'] = torch.from_numpy(input_data['h_a']).float().to(input_data['device'])
        input_data['h_c'] = torch.from_numpy(input_data['h_c']).float().to(input_data['device'])
        forward_out = self.forward(input_data)

        probs = torch.clamp(F.softmax(forward_out['logits'], dim=-1), 0.00001, 0.99999).data  # .values()
        m = torch.distributions.Categorical(probs)
        action = m.sample().type(torch.IntTensor)
        output_data = {'action': action.detach().cpu().numpy().squeeze(), 'logits': forward_out['logits'], 'h_a_': forward_out['h_a_'], 'h_c_': forward_out['h_c_']}

        return output_data

    def get_values(self, input_data):
        self.eval()
        forward_out = self.forward(input_data)

        return forward_out['values']

    def init_hidden(self):
        h = torch.zeros((1, 1, self.settings['dr']), dtype=torch.float32)
        return h
