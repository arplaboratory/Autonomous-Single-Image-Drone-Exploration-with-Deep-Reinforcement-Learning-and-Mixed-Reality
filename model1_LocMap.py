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


class FeaturesExtractor_GT(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_GT, self).__init__()
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

        # self.feat_shape = 32 * 4 * 4
        self.feat_shape = 32 * 7 * 7

    def forward(self, input_data):
        inp = input_data['obstacle_gt'].reshape(-1, 1, self.grid, self.grid)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5


class LocMap_Net(nn.Module):
    def __init__(self, settings):
        super(LocMap_Net, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']
        self.dl = settings['dl']
        self.dr = settings['dr']
        self.map_in = 100

        self.features_RGB = FeaturesExtractor_RGB(settings)
        self.features_GT = FeaturesExtractor_GT(settings)

        self.RGB_shape = self.features_RGB.feat_shape
        self.GT_shape = self.features_GT.feat_shape

        self.feat_shape = self.RGB_shape + self.GT_shape

        # lin
        self.lin = nn.Linear(self.feat_shape, self.dl)

        # gru
        self.gru = nn.GRU(self.dl, self.dr)

        # hidden
        self.hidden = nn.Linear(self.dr, 150)

        # position reconstruction
        # self.lin_map_g = nn.Linear(150, self.map_in)
        # self.debnc_g = nn.GroupNorm(int(self.d1 / 2), self.d1)
        # self.deconv1_g = nn.ConvTranspose2d(in_channels=1, out_channels=self.d1,
        #                                     kernel_size=3, stride=1, padding=0, dilation=2)
        # self.deconv2_g = nn.ConvTranspose2d(in_channels=self.d1, out_channels=self.d2,
        #                                     kernel_size=3, stride=1, padding=0, dilation=1)
        # self.deconv3_g = nn.ConvTranspose2d(in_channels=self.d2, out_channels=self.d3,
        #                                     kernel_size=3, stride=1, padding=0, dilation=1)
        # self.deconv4_g = nn.ConvTranspose2d(in_channels=self.d3, out_channels=1,
        #                                     kernel_size=3, stride=1, padding=0, dilation=1)
        self.lin_map_g = nn.Linear(150, self.map_in)
        self.debnc_g = nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.deconv1_g = nn.ConvTranspose2d(in_channels=1, out_channels=self.d1,
                                            kernel_size=4, stride=2, padding=0, dilation=2)
        self.deconv2_g = nn.ConvTranspose2d(in_channels=self.d1, out_channels=self.d2,
                                            kernel_size=3, stride=1, padding=0, dilation=1)
        self.deconv3_g = nn.ConvTranspose2d(in_channels=self.d2, out_channels=self.d3,
                                            kernel_size=3, stride=1, padding=0, dilation=1)
        self.deconv4_g = nn.ConvTranspose2d(in_channels=self.d3, out_channels=1,
                                            kernel_size=3, stride=1, padding=0, dilation=3)

        # target reconstruction
        # self.lin_map_b = nn.Linear(150, self.map_in)
        # self.debnc_b = nn.GroupNorm(int(self.d1 / 2), self.d1)
        # self.deconv1_b = nn.ConvTranspose2d(in_channels=1, out_channels=self.d1,
        #                                     kernel_size=3, stride=1, padding=0, dilation=2)
        # self.deconv2_b = nn.ConvTranspose2d(in_channels=self.d1, out_channels=self.d2,
        #                                     kernel_size=3, stride=1, padding=0, dilation=1)
        # self.deconv3_b = nn.ConvTranspose2d(in_channels=self.d2, out_channels=self.d3,
        #                                     kernel_size=3, stride=1, padding=0, dilation=1)
        # self.deconv4_b = nn.ConvTranspose2d(in_channels=self.d3, out_channels=1,
        #                                     kernel_size=3, stride=1, padding=0, dilation=1)
        self.lin_map_b = nn.Linear(150, self.map_in)
        self.debnc_b = nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.deconv1_b = nn.ConvTranspose2d(in_channels=1, out_channels=self.d1,
                                            kernel_size=4, stride=2, padding=0, dilation=2)
        self.deconv2_b = nn.ConvTranspose2d(in_channels=self.d1, out_channels=self.d2,
                                            kernel_size=3, stride=1, padding=0, dilation=1)
        self.deconv3_b = nn.ConvTranspose2d(in_channels=self.d2, out_channels=self.d3,
                                            kernel_size=3, stride=1, padding=0, dilation=1)
        self.deconv4_b = nn.ConvTranspose2d(in_channels=self.d3, out_channels=1,
                                            kernel_size=3, stride=1, padding=0, dilation=3)

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_hm = input_data['h_m']

        featuresRGB = self.features_RGB.forward(input_data).reshape(-1, self.RGB_shape)
        featuresGT = self.features_GT.forward(input_data).reshape(-1, self.GT_shape)
        features = torch.cat((featuresRGB, featuresGT), dim=1)

        # lin
        x = F.relu(self.lin(features))

        # gru
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, h_m_ = self.gru(x_, input_hm)

        # hidden
        xh = F.relu(self.hidden(xgru))

        # position reconstruction (G channel)
        xm_g = F.relu(self.lin_map_g(xh))
        xm_g_ = xm_g.reshape(len(xm_g) * len(xm_g[0]), 1, int(self.map_in ** (1 / 2)), int(self.map_in ** (1 / 2)))
        xm1_g = self.debnc_g(F.relu(self.deconv1_g(xm_g_)))
        xm2_g = F.relu(self.deconv2_g(xm1_g))
        xm3_g = F.relu(self.deconv3_g(xm2_g))
        green_ch = F.relu(self.deconv4_g(xm3_g))  # [seq * batch, 1, grid_cells, grid_cells] raw values

        # target reconstruction (B channel)
        xm_b = F.relu(self.lin_map_b(xh))
        xm_b_ = xm_b.reshape(len(xm_b) * len(xm_b[0]), 1, int(self.map_in ** (1 / 2)), int(self.map_in ** (1 / 2)))
        xm1_b = self.debnc_b(F.relu(self.deconv1_b(xm_b_)))
        xm2_b = F.relu(self.deconv2_b(xm1_b))
        xm3_b = F.relu(self.deconv3_b(xm2_b))
        blue_ch = F.relu(self.deconv4_b(xm3_b))  # [seq * batch, 1, grid_cells, grid_cells] raw values

        # map
        green_prob = F.softmax(green_ch.reshape(1, -1), dim=1).reshape(-1, 1, self.settings['grid_cells'], self.settings['grid_cells'])
        blue_prob = F.softmax(blue_ch.reshape(1, -1), dim=1).reshape(-1, 1, self.settings['grid_cells'], self.settings['grid_cells'])
        map_rec = torch.cat((input_data['obstacle_gt'], green_prob, blue_prob), dim=1)

        output_data = {'h_m_': h_m_, 'red_ch': input_data['obstacle_gt'], 'green_ch': green_ch, 'blue_ch': blue_ch, 'map_rec': map_rec}

        return output_data

    @property
    def learnable(self):
        return True

    @property
    def device(self):
        return next(self.parameters()).device

    def init_hidden(self):
        h_m = torch.zeros((1, 1, self.settings['dr']), dtype=torch.float32)
        return h_m

    def get_map(self, input_data):
        self.eval()
        input_data['h_m'] = torch.from_numpy(input_data['h_m']).float().to(input_data['device'])
        input_data['obstacle_gt'] = input_data['obstacle_gt'].unsqueeze(0).unsqueeze(0)
        forward_out = self.forward(input_data)
        red_ch = forward_out['red_ch']
        # red_ch_clamp = torch.clamp(red_ch, min=0., max=1.)  # GroundTruth already between 0 and 1
        green_ch = forward_out['green_ch']
        green_prob = F.softmax(green_ch.reshape(1, -1), dim=1).reshape(-1, 1, self.settings['grid_cells'], self.settings['grid_cells'])
        blue_ch = forward_out['blue_ch']
        blue_prob = F.softmax(blue_ch.reshape(1, -1), dim=1).reshape(-1, 1, self.settings['grid_cells'], self.settings['grid_cells'])
        # map RGB
        map_rec = torch.cat((red_ch, green_prob, blue_prob), dim=1).squeeze().to('cuda: 1')  # [seq * batch, 3, grid_cells, grid_cells]
        output_data = {'h_m_': forward_out['h_m_'], 'map_rec': map_rec}
        return output_data
