import torch
import torch.nn.functional as F
import torch.nn as nn


class FeaturesExtractor(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor, self).__init__()
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

        self.feat_shape = 32 * 8 * 8

    def forward(self, input_data):
        inp = input_data['RGBCamera'].reshape(-1, 3, 84, 84)

        # x1 = self.maxp(inp)
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

        self.features = FeaturesExtractor(settings)
        self.feat_shape = self.features.feat_shape

        # GroupNorm
        self.bn1 = nn.GroupNorm(self.d1, self.d2)

        # Linear
        self.lin = nn.Linear(self.feat_shape, self.dl)

        # GRU
        self.gru = nn.GRU(self.dl, self.dr)

        # hidden
        self.hidden = nn.Linear(self.dr, 100)

        # map reconstruction
        self.lin_map_r = nn.Linear(100, self.map_in)
        self.debnc_r = nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.deconv1_r = nn.ConvTranspose2d(in_channels=1, out_channels=self.d1,
                                            kernel_size=3, stride=1, padding=0, dilation=2)
        self.deconv2_r = nn.ConvTranspose2d(in_channels=self.d1, out_channels=self.d2,
                                            kernel_size=3, stride=1, padding=0, dilation=1)
        self.deconv3_r = nn.ConvTranspose2d(in_channels=self.d2, out_channels=self.d3,
                                            kernel_size=3, stride=1, padding=0, dilation=1)
        self.deconv4_r = nn.ConvTranspose2d(in_channels=self.d3, out_channels=1,
                                            kernel_size=3, stride=1, padding=0, dilation=1)

        # position reconstruction
        self.lin_map_g = nn.Linear(100, self.map_in)
        self.debnc_g = nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.deconv1_g = nn.ConvTranspose2d(in_channels=1, out_channels=self.d1,
                                            kernel_size=3, stride=1, padding=0, dilation=2)
        self.deconv2_g = nn.ConvTranspose2d(in_channels=self.d1, out_channels=self.d2,
                                            kernel_size=3, stride=1, padding=0, dilation=1)
        self.deconv3_g = nn.ConvTranspose2d(in_channels=self.d2, out_channels=self.d3,
                                            kernel_size=3, stride=1, padding=0, dilation=1)
        self.deconv4_g = nn.ConvTranspose2d(in_channels=self.d3, out_channels=1,
                                            kernel_size=3, stride=1, padding=0, dilation=1)

        # target reconstruction
        self.lin_map_b = nn.Linear(100, self.map_in)
        self.debnc_b = nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.deconv1_b = nn.ConvTranspose2d(in_channels=1, out_channels=self.d1,
                                            kernel_size=3, stride=1, padding=0, dilation=2)
        self.deconv2_b = nn.ConvTranspose2d(in_channels=self.d1, out_channels=self.d2,
                                            kernel_size=3, stride=1, padding=0, dilation=1)
        self.deconv3_b = nn.ConvTranspose2d(in_channels=self.d2, out_channels=self.d3,
                                            kernel_size=3, stride=1, padding=0, dilation=1)
        self.deconv4_b = nn.ConvTranspose2d(in_channels=self.d3, out_channels=1,
                                            kernel_size=3, stride=1, padding=0, dilation=1)

        # output
        self.output = nn.Linear(100, self.settings['output_a'])

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_ha = input_data['h_a']

        features = self.features.forward(input_data)

        x = F.relu(self.lin(self.bn1(F.relu(features)).view(-1, self.feat_shape)))
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, h_a_ = self.gru(x_, input_ha)
        xh = F.relu(self.hidden(xgru))

        logits = self.output(xh)

        with torch.autograd.set_detect_anomaly(True):

            # map reconstruction (R channel)
            xm_r = F.relu(self.lin_map_r(xh))
            xm_r_ = xm_r.reshape(len(xm_r) * len(xm_r[0]), 1, int(self.map_in**(1/2)), int(self.map_in**(1/2)))
            xm1_r = self.debnc_r(F.relu(self.deconv1_r(xm_r_)))
            xm2_r = F.relu(self.deconv2_r(xm1_r))
            xm3_r = F.relu(self.deconv3_r(xm2_r))
            red_ch = F.relu(self.deconv4_r(xm3_r))  # [seq * batch, 1, grid_cells, grid_cells]

            # position reconstruction (G channel)
            xm_g = F.relu(self.lin_map_g(xh))
            xm_g_ = xm_g.reshape(len(xm_g) * len(xm_g[0]), 1, int(self.map_in**(1/2)), int(self.map_in**(1/2)))
            xm1_g = self.debnc_g(F.relu(self.deconv1_g(xm_g_)))
            xm2_g = F.relu(self.deconv2_g(xm1_g))
            xm3_g = F.relu(self.deconv3_g(xm2_g))
            green_ch = F.relu(self.deconv4_g(xm3_g))  # [seq * batch, 1, grid_cells, grid_cells]

            # target reconstruction (B channel)
            xm_b = F.relu(self.lin_map_b(xh))
            xm_b_ = xm_b.reshape(len(xm_b) * len(xm_b[0]), 1, int(self.map_in**(1/2)), int(self.map_in**(1/2)))
            xm1_b = self.debnc_b(F.relu(self.deconv1_b(xm_b_)))
            xm2_b = F.relu(self.deconv2_b(xm1_b))
            xm3_b = F.relu(self.deconv3_b(xm2_b))
            blue_ch = F.relu(self.deconv4_b(xm3_b))  # [seq * batch, 1, grid_cells, grid_cells]

        return logits, h_a_, red_ch, green_ch, blue_ch


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

        # self.features = FeaturesExtractorResNet(settings)
        self.features = FeaturesExtractor(settings)
        self.feat_shape = self.features.feat_shape

        # GroupNorm
        self.bn1 = nn.GroupNorm(self.d1, self.d2)

        # Linear
        self.lin = nn.Linear(self.feat_shape, self.dl)

        # GRU
        self.gru = nn.GRU(self.dl, self.dr)

        # hidden
        self.hidden = nn.Linear(self.dr, 100)

        # output
        self.output = nn.Linear(100, 1)

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_hc = input_data['h_c']

        features = self.features.forward(input_data)

        x = F.relu(self.lin(self.bn1(F.relu(features)).view(-1, self.feat_shape)))
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, h_c_ = self.gru(x_, input_hc)

        xh = F.relu(self.hidden(xgru))

        values = self.output(xh)

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
        logits, h_a_, red_ch, green_ch, blue_ch = self.actor(input_data)
        values, h_c_ = self.critic(input_data)
        output_data = {'logits': logits, 'values': values, 'h_a_': h_a_, 'h_c_': h_c_,
                       'red_ch': red_ch, 'green_ch': green_ch, 'blue_ch': blue_ch}

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
        output_data = {'action': action.detach().cpu().numpy().squeeze(), 'logits': forward_out['logits'], 'h_a_': forward_out['h_a_'],  'h_c_': forward_out['h_c_']}

        return output_data

    def get_logits(self, input_data):
        self.train()
        forward_out = self.forward(input_data)

        return forward_out['logits']

    def get_values(self, input_data):
        self.eval()
        forward_out = self.forward(input_data)

        return forward_out['values']

    def init_hidden(self):
        h = torch.zeros((1, 1, self.settings['dl']), dtype=torch.float32)
        return h

    def get_map(self, input_data):
        self.eval()
        forward_out = self.forward(input_data)
        red_ch = forward_out['red_ch']
        red_ch_clamp = torch.clamp(red_ch, min=0., max=1.)
        green_ch = forward_out['green_ch']
        green_prob = F.softmax(green_ch.reshape(1, -1), dim=1).reshape(-1, 1, self.settings['grid_cells'], self.settings['grid_cells'])
        blue_ch = forward_out['blue_ch']
        blue_ch_clamp = torch.clamp(blue_ch, min=0., max=1.)

        # map RGB
        map_rec = torch.cat((red_ch_clamp, green_prob, blue_ch_clamp), dim=1)  # [seq * batch, 3, grid_cells, grid_cells]

        return map_rec.squeeze().to('cuda: 1')

