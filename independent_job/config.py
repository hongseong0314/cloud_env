from ml_collections import config_dict

def base_config():
    cfg = config_dict.ConfigDict()
    cfg.machines_number = 5
    cfg.jobs_len = 10
    cfg.epoch = 100
    cfg.n_episode = 6
    return cfg

def matrix_config():
    cfg = base_config()
    cfg.model_params = {
                        'embedding_dim': 64,
                        'sqrt_embedding_dim': 64**(1/2),
                        'encoder_layer_num': 3,
                        'qkv_dim': 8,
                        'sqrt_qkv_dim': 8**(1/2),
                        'head_num': 8,
                        'logit_clipping': 10,
                        'ff_hidden_dim': 64,

                        'nT':5,
                        'nM':2,

                        'depth_hidden_dim':4,
                        'depth__init':(1/2)**(1/2),
                        'FC_init':(1/4)**(1/2),

                        'ms_hidden_dim': 5,
                        'ms_layer1_init': (1/2)**(1/2),
                        'ms_layer2_init': (1/5)**(1/2),

                        'save_path' : None,
                        'load_path' : None,
                        'skip':False,
                    }
    cfg.optimizer_params = {
                        'optimizer': {
                            'lr': 1e-4,
                            'weight_decay': 1e-6
                        },
                        'scheduler': {
                            'milestones': [101, 151],
                            'gamma': 0.1
                        }
                    }
    return cfg

def fit_config():
    cfg = base_config()
    cfg.model_params = {
                        'nT':5,
                        'nM':2,
                        'save_path' : None,
                        'load_path' : None,
                    }
    cfg.optimizer_params = {
                        'optimizer': {
                            'lr': 1e-3,
                            'weight_decay': 1e-5
                        },
                        'scheduler': {
                            'milestones': [101, 151],
                            'gamma': 0.1
                        }
                    }

    return cfg
