import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy


def create_res_block(dim):
    bnc1 = torch.nn.GroupNorm(int(dim / 2), dim)
    conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
    bnc2 = torch.nn.GroupNorm(int(dim / 2), dim)
    conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
    return bnc1, conv1, bnc2, conv2


def create_big_block(in_dim, out_dim):
    bnc1 = torch.nn.GroupNorm(int(in_dim / 2), in_dim)
    conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0)
    max1 = nn.MaxPool2d(3, stride=2)

    bnc2, conv2, bnc3, conv3 = create_res_block(out_dim)
    bnc4, conv4, bnc5, conv5 = create_res_block(out_dim)
    return bnc1, bnc2, bnc3, bnc4, bnc5, conv1, conv2, conv3, conv4, conv5, max1


def res_block(x, conv1, bnc1, conv2, bnc2):
    x1 = conv1(F.relu(bnc1(x)))
    x1 = conv2(F.relu(bnc2(x1)))
    return x + x1


def big_block(x, conv, bnc, max_p):
    x1 = max_p(conv[0](bnc[0](x)))
    x1 = res_block(x1, conv[1], bnc[1], conv[2], bnc[2])
    x1 = res_block(x1, conv[3], bnc[3], conv[4], bnc[4])
    return x1


# ################################-Features-Extractors-###############################################################
class FeaturesExtractorResNet_RGB(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractorResNet_RGB, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']
        self.dl = settings['dl']
        self.dr = settings['dr']
        self.feat_shape = 32 * 7 * 7

        # 1
        self.bnc1, self.bnc2, self.bnc3, self.bnc4, self.bnc5, \
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, \
            self.max1 = create_big_block(self.channels, self.d1)

        self.bnc_1 = [self.bnc1, self.bnc2, self.bnc3, self.bnc4, self.bnc5]
        self.conv_1 = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

        # 2
        self.bnc6, self.bnc7, self.bnc8, self.bnc9, self.bnc10, \
            self.conv6, self.conv7, self.conv8, self.conv9, self.conv10, \
            self.max2 = create_big_block(self.d1, self.d2)

        self.bnc_2 = [self.bnc6, self.bnc7, self.bnc8, self.bnc9, self.bnc10]
        self.conv_2 = [self.conv6, self.conv7, self.conv8, self.conv9, self.conv10]

        # 3
        self.bnc11, self.bnc12, self.bnc13, self.bnc14, self.bnc15, \
            self.conv11, self.conv12, self.conv13, self.conv14, self.conv15, \
            self.max3 = create_big_block(self.d2, self.d2)

        self.bnc_3 = [self.bnc11, self.bnc12, self.bnc13, self.bnc14, self.bnc15]
        self.conv_3 = [self.conv11, self.conv12, self.conv13, self.conv14, self.conv15]

    def forward(self, input_data):
        inp = input_data['RGBCamera'].reshape(-1, 3, 84, 84)

        x1 = big_block(inp, self.conv_1, self.bnc_1, self.max1)
        x2 = big_block(x1, self.conv_2, self.bnc_2, self.max2)
        features = big_block(x2, self.conv_3, self.bnc_3, self.max3)

        return features

    @property
    def device(self):
        return next(self.parameters()).device


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

    @property
    def device(self):
        return next(self.parameters()).device


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

    @property
    def device(self):
        return next(self.parameters()).device


class FeaturesExtractor_Grid(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_Grid, self).__init__()
        self.settings = settings
        self.grid = settings['grid_cells']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.d1,
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

        self.feat_shape = 32 * 7 * 7

    def forward(self, input_data):
        inp = input_data['bel'].reshape(-1, 3, self.grid, self.grid).to(self.device)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5

    @property
    def device(self):
        return next(self.parameters()).device


class FeaturesExtractor_Ego(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_Ego, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.width = 21  # TODO
        self.height = 11
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.d1,
                               kernel_size=3, stride=2, padding=1)
        self.bnc1 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d1,
                               kernel_size=3, stride=1, padding=1)
        self.bnc2 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv3 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=1)
        self.bnc3 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)
        self.conv4 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=0)
        self.bnc4 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)

        self.feat_shape = 32 * 9 * 4

    def forward(self, input_data):
        inp = input_data['ego_target'].reshape(-1, self.channels, self.width, self.height)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5

    @property
    def device(self):
        return next(self.parameters()).device
######################################################################################################################


# ###################################-Localization+Mapping-###########################################################
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

        # self.feat_shape = self.RGB_shape + self.GT_shape + 1  # 1 action
        self.feat_shape = self.RGB_shape + self.GT_shape

        # lin
        self.lin = nn.Linear(self.feat_shape, self.dl)

        # gru
        self.gru = nn.GRU(self.dl, self.dr)

        # hidden
        self.hidden = nn.Linear(self.dr, 150)

        # position reconstruction
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
        self.lin_map_b = nn.Linear(150, self.map_in)
        self.debnc_b = nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.deconv1_b = nn.ConvTranspose2d(in_channels=1, out_channels=self.d1,
                                            kernel_size=(3, 3), stride=(1, 2), padding=(0, 2), dilation=(1, 2))
        self.deconv2_b = nn.ConvTranspose2d(in_channels=self.d1, out_channels=self.d3,
                                            kernel_size=(2, 3), stride=(1, 1), padding=(2, 1), dilation=(1, 1))
        self.deconv3_b = nn.ConvTranspose2d(in_channels=self.d3, out_channels=self.d3,
                                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.deconv4_b = nn.ConvTranspose2d(in_channels=self.d3, out_channels=1,
                                            kernel_size=3, stride=1, padding=0, dilation=1)

    def forward(self, input_data, gt=True):
        input_shape = input_data['RGBCamera'].shape
        input_hm = input_data['h_m']
        input_action = input_data['action']

        featuresRGB = self.features_RGB.forward(input_data).reshape(-1, self.RGB_shape)
        featuresGT = self.features_GT.forward(input_data).reshape(-1, self.GT_shape)
        # features = torch.cat((featuresRGB, featuresGT, input_action.reshape(-1, 1)), dim=1)
        features = torch.cat((featuresRGB, featuresGT), dim=1)

        # lin
        x = F.relu(self.lin(features))

        # gru
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, h_m_ = self.gru(x_, input_hm)

        # hidden
        xh = F.relu(self.hidden(xgru))

        # ################-new_geo_hat-#########################
        # position global (G channel)
        xm_g = F.relu(self.lin_map_g(xh))
        xm_g_ = xm_g.reshape(len(xm_g) * len(xm_g[0]), 1, int(self.map_in ** (1 / 2)), int(self.map_in ** (1 / 2)))
        xm1_g = self.debnc_g(F.relu(self.deconv1_g(xm_g_)))
        xm2_g = F.relu(self.deconv2_g(xm1_g))
        xm3_g = F.relu(self.deconv3_g(xm2_g))
        geo_position_hat = F.relu(self.deconv4_g(xm3_g))  # [-1, 1, grid_cells, grid_cell] raw values
        ##################################################

        # ################-ego_hat-#######################
        # target ego (B channel)
        xm_b = F.relu(self.lin_map_b(xh))
        xm_b_ = xm_b.reshape(len(xm_b) * len(xm_b[0]), 1, int(self.map_in ** (1 / 2)), int(self.map_in ** (1 / 2)))
        xm1_b = self.debnc_b(F.relu(self.deconv1_b(xm_b_)))
        xm2_b = F.relu(self.deconv2_b(xm1_b))
        xm3_b = F.relu(self.deconv3_b(xm2_b))
        blue_ch = F.relu(self.deconv4_b(xm3_b))
        blue_ch = torch.clamp(blue_ch, min=0., max=1.)  # [-1, 1, 11, 21]

        # position ego (G channel)
        green_ch = torch.zeros_like(blue_ch)  # [-1, 1, 11, 21]
        green_ch[:, :, 10, 10] = 1  # for visualization

        # obstacle ego (R channel)
        red_ch = torch.zeros_like(blue_ch)  # [-1, 1, 11, 21]

        # ego_map
        ego_hat = torch.cat((red_ch, green_ch, blue_ch), dim=1).to(self.device)  # [-1, 3, 11, 21]
        ego_hat = torch.clamp(ego_hat, min=0., max=1.)
        ##################################################

        input_data['target_hat'] = input_data['target_hat'].reshape(-1, 1, self.settings['grid_cells'], self.settings['grid_cells'])

        # tracker position
        if gt:
            t_pos = input_data['Tracker_position'].to(self.device)
        else:
            green_prob = F.softmax(geo_position_hat.reshape(1, -1), dim=1).reshape(-1, self.settings['grid_cells'], self.settings['grid_cells'])
            t_pos = torch.zeros(green_prob.size(0), 2).to(self.device)
            for i in range(len(t_pos)):
                t_pos[i, 0] = int(int(torch.argmax(green_prob[i])) / self.settings['grid_cells'])
                t_pos[i, 1] = int(torch.argmax(green_prob[i])) % self.settings['grid_cells']

        # TODO: replace gt_position with estimate_position
        geo_hat = torch.cat((input_data['obstacle_gt'], input_data['tracker_gt'], input_data['target_hat']), dim=1)  # [-1, 3, grid_cells, grid_cells]

        # geo_hat t = geo_hat t-1 [n_cell * n_cell] - ego_map t [11 * 21]
        new_geo_hat = torch.clamp((self.get_visibility(geo_hat, ego_hat, t_pos, input_data['yaw'].to(self.device))), min=0., max=1.)
        new_target_hat = torch.clamp(new_geo_hat[:, 2], min=0., max=1.)  # [-1 , grid_cells, grid_cells]

        # tmp
        input_data['target_gt'] = input_data['target_gt'].reshape(-1, 1, self.settings['grid_cells'], self.settings['grid_cells'])  # TODO
        geo_target = torch.cat((input_data['obstacle_gt'], input_data['tracker_gt'], input_data['target_gt']), dim=1)  # [-1, 3, grid_cells, grid_cells]

        output_data = {'h_m_': h_m_, 'target_hat_': new_target_hat,  # persistence
                       # loss computation
                       'ego_hat': ego_hat,                           # ego map estimation
                       'geo_position_hat': geo_position_hat,         # position estimation
                       # te_model input
                       'geo_hat': new_geo_hat,                       # visibility map
                       'geo_target': geo_target}                     # tmp

        return output_data

    @property
    def device(self):
        return next(self.parameters()).device

    def get_visibility(self, gt_grid, ego_grid_hat, tracker_p, yaw):

        # turn back
        angle = torch.from_numpy(np.array(np.radians(yaw.cpu().numpy()))).to(self.device)
        theta = torch.zeros(gt_grid.size(0), 2, 3).to(self.device)
        if angle.nelement() > 1:
            for i in range(len(angle)):
                theta[i, :, :2] = torch.tensor([[torch.cos(angle[i]).to(self.device), torch.tensor([-1.0]).to(self.device) * torch.sin(angle[i]).to(self.device)],
                                                [torch.sin(angle[i]).to(self.device), torch.cos(angle[i]).to(self.device)]])
        else:
            theta[:, :, :2] = torch.tensor([[torch.cos(angle).to(self.device), torch.tensor([-1.0]).to(self.device) * torch.sin(angle).to(self.device)],
                                                    [torch.sin(angle).to(self.device), torch.cos(angle).to(self.device)]])

        ego_grid_hat = F.pad(input=ego_grid_hat[:], pad=[0, 0, 0, ego_grid_hat.size(-1) - ego_grid_hat.size(-2)], mode='constant', value=0)  # [-1, 3, 21, 21]
        grid = F.affine_grid(theta, ego_grid_hat.size())
        x_trans = F.grid_sample(ego_grid_hat, grid)
        # grid_affine = F.pad(input=x_trans[:], pad=[0, 0, 0, x_trans.size(-1) - x_trans.size(-2)], mode='constant', value=0)
        grid_affine = x_trans

        # thresholding
        grid_affine[:, 2] = torch.where(grid_affine[:, 2] < 0.5, torch.Tensor([0]).to(self.device), grid_affine[:, 2])
        grid_affine[:, 2] = torch.where(grid_affine[:, 2] > 0.5, torch.Tensor([1]).to(self.device), grid_affine[:, 2])
        grid_affine[:, 1] = torch.where(grid_affine[:, 1] < 0.5, torch.Tensor([0]).to(self.device), grid_affine[:, 1])
        grid_affine[:, 1] = torch.where(grid_affine[:, 1] > 0.5, torch.Tensor([1]).to(self.device), grid_affine[:, 1])

        # geocentric grid reconstruction
        rec_grid = copy.copy(gt_grid)  # [-1, 3, grid_cells, grid_cells]
        # target_pos = torch.where(rec_grid[:, 2] > 0)  # TODO target pos
        pad = int(grid_affine.size(-1) / 2) + 1
        rec_grid = F.pad(input=rec_grid[:], pad=[pad, pad, pad, pad], mode='constant', value=0)
        for i in range(tracker_p.size(0)):
            tracker_pos = (int(tracker_p[i, 0] + pad), int(tracker_p[i, 1] + pad))
        tmp = rec_grid[:, 2, tracker_pos[0] - pad + 1: tracker_pos[0] + pad, tracker_pos[1] - pad + 1: tracker_pos[1] + pad]
        diff = tmp - grid_affine[:, 2]
        rec_grid[:, 2, tracker_pos[0] - pad + 1: tracker_pos[0] + pad, tracker_pos[1] - pad + 1: tracker_pos[1] + pad] = diff  # -= grid_affine[:, 2]

        rec_grid = rec_grid[:, :, pad: -pad, pad: -pad]
        rec_grid[:, 2] = torch.where(rec_grid[:, 0] > 0, torch.Tensor([0]).to(self.device), rec_grid[:, 2])  # blue channel = 0 where red channel != 0
        rec_grid[:, 2] = torch.where(rec_grid[:, 1] > 0, torch.Tensor([0]).to(self.device), rec_grid[:, 2])  # blue channel = 0 where green channel != 0
        # rec_grid[:, 2][target_pos] = torch.Tensor([1.].to(self.device)  # TODO target pos

        return rec_grid
######################################################################################################################


# ###################################-Tracking+Exploration-###########################################################
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

        if self.settings['resnet']:
            self.features_RGB = FeaturesExtractorResNet_RGB(settings)
        else:
            self.features_RGB = FeaturesExtractor_RGB(settings)
        # self.features_Grid = FeaturesExtractor_Grid(settings)

        self.RGB_shape = self.features_RGB.feat_shape
        # self.Grid_shape = self.features_Grid.feat_shape

        # self.feat_shape = self.RGB_shape + self.Grid_shape
        self.feat_shape = self.RGB_shape

        # Linear
        self.lin1 = nn.Linear(self.feat_shape, 750)
        self.lin2 = nn.Linear(750, self.dl)

        # GRU
        self.gru = nn.GRU(self.dl, self.dr)

        # hidden
        self.hidden0 = nn.Linear(self.dr, 512)
        self.hidden1 = nn.Linear(512, 256)
        self.hidden2 = nn.Linear(256, 128)

        # output
        self.output = nn.Linear(128, self.settings['output_a'])

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_ha = input_data['h_a']

        featuresRGB = self.features_RGB.forward(input_data).reshape(-1, self.RGB_shape)
        # featuresGrid = self.features_Grid.forward(input_data).reshape(-1, self.Grid_shape)

        # features = torch.cat((featuresRGB, featuresGrid), dim=1)

        # lin
        x1 = F.relu(self.lin1(featuresRGB))
        x2 = F.relu(self.lin2(x1))

        # gru
        x_ = x2.view(input_shape[0], -1, self.dl)
        xgru, h_a_ = self.gru(x_, input_ha)

        # hidden
        xh0 = F.relu(self.hidden0(xgru))
        xh1 = F.relu(self.hidden1(xh0))
        xh2 = F.relu(self.hidden2(xh1))

        logits = self.output(xh2)
        # logits = output_data[:, :, :-1]
        # fov = output_data[:, :, -1].unsqueeze(-1)
        # print(torch.sigmoid(fov))
        fov = logits
        return logits, fov, h_a_


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

        # self.features_RGB = FeaturesExtractor_RGB(settings)
        self.features_Grid = FeaturesExtractor_Grid(settings)
        self.features_Ego = FeaturesExtractor_Ego(settings)

        # self.RGB_shape = self.features_RGB.feat_shape
        self.Grid_shape = self.features_Grid.feat_shape
        self.Ego_shape = self.features_Ego.feat_shape

        # self.feat_shape = self.RGB_shape + self.Grid_shape
        self.feat_shape = self.Grid_shape + self.Ego_shape

        # Linear
        # self.lin_grids = nn.Linear(self.feat_shape, 750)
        self.lin_yaw = nn.Linear(1, 100)
        self.lin_dis_angle_hit = nn.Linear(3, 300)

        self.lin = nn.Linear(self.feat_shape + 100 + 300, self.dl)

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
        input_yaw = input_data['yaw'].reshape(-1, 1)
        distance = input_data['distance'].reshape(-1, 1)
        angle = input_data['angle'].reshape(-1, 1)
        hit = input_data['hit'].reshape(-1, 1)

        dis_angle_hit = torch.cat((distance, angle, hit), dim=1)

        # featuresRGB = self.features_RGB.forward(input_data).reshape(-1, self.RGB_shape)
        featuresGrid = self.features_Grid.forward(input_data).reshape(-1, self.Grid_shape)
        featuresEgo = self.features_Ego.forward(input_data).reshape(-1, self.Ego_shape)

        features = torch.cat((featuresGrid, featuresEgo), dim=1)

        # lin
        # x_grids = F.relu(self.lin_grids(features))
        x_yaw = F.relu(self.lin_yaw(input_yaw))
        x_dist_angle_hit = F.relu(self.lin_dis_angle_hit(dis_angle_hit))
        x_cat = torch.cat((features, x_yaw, x_dist_angle_hit), dim=1)
        x = F.relu(self.lin(x_cat))

        # gru
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, h_c_ = self.gru(x_, input_hc)

        # hidden
        xh1 = F.relu(self.hidden1(xgru))
        xh2 = F.relu(self.hidden2(xh1))

        values = self.output(xh2)

        return values, h_c_
######################################################################################################################


# #####################################-Impala_Net-###################################################################
class IMPALA_Net(nn.Module):
    def __init__(self, settings):
        super(IMPALA_Net, self).__init__()
        self.settings = settings
        # self.lm_net = LocMap_Net(settings)
        self.actor = ActorIMPALA(settings)
        self.critic = CriticIMPALA(settings)

    @property
    def learnable(self):
        return True

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_data):
        # lm_out = self.lm_net(input_data)
        # visibility = copy.copy(lm_out['geo_target']).detach()  # TODO use geo_hat
        # input_data['visibility_map'] = visibility
        input_data['visibility_map'] = torch.cat((input_data['obstacle_gt'], input_data['tracker_gt'],
                                                  input_data['target_gt']), dim=1)
        logits, fov, h_a_ = self.actor(input_data)
        values, h_c_ = self.critic(input_data)
        # output_data = {'logits': logits, 'values': values,                                             # loss computation
        #                'h_a_': h_a_, 'h_c_': h_c_,                                                     # persistence
        #                'h_m_': lm_out['h_m_'], 'target_hat_': lm_out['target_hat_'],                   # persistence
        #                'ego_hat': lm_out['ego_hat'], 'geo_position_hat': lm_out['geo_position_hat'],   # loss computation
        #                'geo_hat': lm_out['geo_hat']}                                                   # visualization
        output_data = {'logits': logits, 'values': values,
                       'h_a_': h_a_, 'h_c_': h_c_,
                       'fov': fov}

        return output_data

    def choose_action(self, input_data):
        self.eval()
        input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)
        input_data['h_a'] = torch.from_numpy(input_data['h_a']).float().to(input_data['device'])
        input_data['h_c'] = torch.from_numpy(input_data['h_c']).float().to(input_data['device'])
        # input_data['h_m'] = torch.from_numpy(input_data['h_m']).float().to(input_data['device'])

        # input_data['target_hat'] = torch.from_numpy(input_data['target_hat']).float().to(input_data['device'])

        input_data['obstacle_gt'] = input_data['obstacle_gt'].unsqueeze(0).unsqueeze(0)
        input_data['tracker_gt'] = input_data['tracker_gt'].unsqueeze(0).unsqueeze(0)
        input_data['target_gt'] = input_data['target_gt'].unsqueeze(0).unsqueeze(0)

        forward_out = self.forward(input_data)

        probs = torch.clamp(F.softmax(forward_out['logits'], dim=-1), 0.00001, 0.99999).data  # .values()
        m = torch.distributions.Categorical(probs)
        action = m.sample().type(torch.IntTensor)

        # gph = F.softmax(forward_out['geo_position_hat'].reshape(1, -1), dim=1).reshape(-1, 1, self.settings['grid_cells'], self.settings['grid_cells'])

        # output_data = {'action': action.detach().cpu().numpy().squeeze(), 'logits': forward_out['logits'].detach().squeeze().cpu().numpy(),
        #                'h_a_': forward_out['h_a_'], 'h_c_': forward_out['h_c_'],
        #                'h_m_': forward_out['h_m_'], 'target_hat_': forward_out['target_hat_'],
        #                'ego_hat': forward_out['ego_hat'].squeeze(),
        #                'geo_hat': forward_out['geo_hat'].squeeze(),
        #                'geo_position_hat': gph}

        output_data = {'action': action.detach().cpu().numpy().squeeze(), 'logits': forward_out['logits'].detach().squeeze().cpu().numpy(),
                       'h_a_': forward_out['h_a_'], 'h_c_': forward_out['h_c_']}

        return output_data

    def get_values(self, input_data):
        self.eval()
        forward_out = self.forward(input_data)

        return forward_out['values']

    def init_hidden(self):
        h = torch.zeros((1, 1, self.settings['dr']), dtype=torch.float32)
        return h
######################################################################################################################
