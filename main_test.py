if __name__ == '__main__':
    from gym import spaces
    from model_1CA import IMPALA_Net
    from my_agent import MyAgent
    from live_plot import GridPlot
    from live_plot import RewardPlot
    from live_plot import MaprecPlot
    from isarsocket.simple_socket import SimpleSocket
    import torch.multiprocessing as mp
    import numpy as np
    import random
    import torch
    import json
    import sys
    import os

    eval_mode = True
    TD3 = False
    IMPALA = True
    hot_reload = False

    grid_plot = False
    reward_plot = False
    maprec_plot = False
    plot_list = [grid_plot, reward_plot, maprec_plot]
    n_plot = plot_list.count(True)

    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Path for Models
    pathname = os.path.dirname(sys.argv[0])
    abs_path = os.path.abspath(pathname)

    state_dict = ''

    json_paths = [['{}/settings/IMPALA/settings_coord.json'.format(abs_path)],
                  # f = open('settings/IMPALA/settings.json')
                  ['{}/settings/IMPALA/settings_coord.json'.format(abs_path),
                   '{}/settings/IMPALA/settings_target.json'.format(
                       abs_path)]]  # f = open('settings/IMPALA/settings.json')
    f = open('settings/IMPALA/settings_coord.json')  # f = open('settings/IMPALA/settings.json')
    algorithm = 'IMPALA'
    net = IMPALA_Net
    action_space = spaces.Discrete(8)
    n_actions = 8
    assert IMPALA != TD3

    json_settings = json.load(f)

    print('MAIN', os.getpid())

    mp.set_start_method('spawn')

    # [sequence, batch, features]
    seq_len = 50  # 50
    memory_len = 1500  # 10000
    episode_length = 1800  # 1800
    eval_episodes = 20
    N_Agents = 8
    batch_size = 24  # 24
    main_to_agent_address = 6734
    agent_address_list = [('127.0.0.1', main_to_agent_address + i) for i in range(N_Agents + 2)]
    plot_address = 10734

    RGB_width = json_settings['sensor_settings']['RGBCamera']['width']
    RGB_height = json_settings['sensor_settings']['RGBCamera']['height']
    grid_cells = json_settings['sensor_settings']['OccupancyGrid']['size_x']

    settings = {
        'model_param': {
            'dl_1': 400,
            'dl_2': 300,
            'd1': 16,
            'd2': 32,
            'd3': 64,
            'dl': 256,
            'dr': 256,
            'channels': 3,
            'width': RGB_width,
            'height': RGB_height,
            'RGB_state_space': np.array([RGB_width, RGB_height, 3]),
            'state_space': 2,
            'init_w': 3e-3,
            'output_a': n_actions,
            'exp_distance': 50,
            'max_distance': 20,
            'TD3': TD3,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'grid_cells': grid_cells,
            'resnet': False,
        },
        'agent_gpu_id': 1,
        'update_steps': 100,
        'env_track': 'unrealisar-track-v0',
        'env_exp': 'unrealisar-exp-v0',
        'env_name': 'unrealisar-lg-v0',
        'multi_mod': True,
        'unreal': {
            'start_port_test': 9734,
            'start_port_train': 9736,
            'json_paths': json_paths,
            'render': True,
            'address': '10.8.0.2'
        },
        'test': True,
        'fov_gt_test': False,
        'eval_mode': eval_mode,
        'eval_episodes': eval_episodes,
        'seq_len': seq_len,
        'memory_len': memory_len,
        'action_space': action_space,
        'action_space_noise': spaces.Box(low=-1, high=1, shape=(1,)),
        'workstation_name': 'Idra',
        'main_to_agent_address': ('127.0.0.1', main_to_agent_address),
        'first_address': ('127.0.0.1', 7734),
        'learner_address': ('127.0.0.1', 8734),
        'episode_length': episode_length,
        'warmup': seq_len + batch_size - 1 + int((seq_len + batch_size - 1) / episode_length) * (seq_len - 1),
        'on_policy': False,
        'learner_rank': 0,
        'lr_actor': 0.0001,
        'lr_critic': 0.001,
        'lr_impala': 0.0002,
        'batch_size': batch_size,
        'learner_gpu_id': 0,
        'memory_address': ('127.0.0.1', 7734 + N_Agents),
        'num_agents': N_Agents,
        'tau': 0.01,
        'gamma': 0.99,
        'RMSprop_eps': 0.1,
        'algorithm': algorithm,
        'baseline_cost': 0.5,
        'entropy_cost': 0.001,  # 0.001
        'navigation_cost': 1,
        'obstacle_rec_cost': 1,
        'tracker_rec_cost': 1e2,
        'target_rec_cost': 1e2,
        'TD3': TD3,
        'IMPALA': IMPALA,
        'update_settings': {
            'batch_size': batch_size,
            'min_new_samples': 50 * N_Agents,
            'mode': 'newest',
            'data_keys': {
                'RGBCamera': seq_len,
                'action_track': seq_len,
                'action_exp': seq_len,
                'reward_track': seq_len,
                'reward_exp': seq_len,
                'next_state': 1,
                'done_track': seq_len,
                'done_exp': seq_len,
                'track_h_a': 1,
                'exp_h_a': 1,
                'fov_h_a': 1,
                'track_h_c': 1,
                'exp_h_c': 1,
                'logits_track': seq_len,
                'logits_exp': seq_len,
                'yaw': seq_len,
                'ego_target': seq_len,
                'angle': seq_len,
                'distance': seq_len,
                'hit': seq_len,
                'fov_gt': seq_len,
                'bel': seq_len,
                'track_flag': seq_len
            }
        },
        'agents_fps_array': mp.Array('d', np.zeros(N_Agents)),
        'agent_rank': mp.Value('i', 0),
        'shared_counter': mp.Value('i', 0),
        'plot_param': {
            'plot_address': ('127.0.0.1', plot_address),
            'plot_list': plot_list,
            'n_plot': n_plot,
            'rew_len': episode_length,
            'eval_mode': eval_mode
        },
        'grid_cells': grid_cells,
        'obstacle_ch_weight': 1,
        'tracker_ch_weight': 1,
        'target_ch_weight': 2,
        'state_dict': state_dict
    }

    settings['test'] = False

    plots = []
    if grid_plot:
        grid_mp = GridPlot(settings=settings['plot_param'])
        plots.append(grid_mp)

    if reward_plot:
        reward_mp = RewardPlot(settings=settings['plot_param'])
        plots.append(reward_mp)

    if maprec_plot:
        maprec_mp = MaprecPlot(settings=settings['plot_param'])
        plots.append(maprec_mp)

    [plot_mp.start() for plot_mp in plots]

    gnet = net(settings=settings['model_param'])
    gnet.load_state_dict(torch.load('C:\\Users\\Idra\\Documents\\ActiveTracking\\NYU_models\\best_model_2.pt'))
    #gnet.load_state_dict(torch.load('/Users/alessandro/Desktop/Exploration/models/best_model_2.pt', map_location=torch.device('cpu')))
    test_agent = MyAgent(settings, gnet)
    test_agent.run()

    [plot_mp.join() for plot_mp in plots]
