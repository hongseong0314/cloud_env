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
                        'encoder_layer_num': 6,
                        'qkv_dim': 8,
                        'sqrt_qkv_dim': 8**(1/2),
                        'head_num': 8,
                        'logit_clipping': 10,
                        'ff_hidden_dim': 64,

                        'eval_type': 'softmax',
                        'nT':5,
                        'nM':2,

                        'depth_hidden_dim':4,
                        'depth__init':(1/2)**(1/2),
                        'FC_init':(1/4)**(1/2),

                        'ms_hidden_dim': 6,
                        'ms_layer1_init': (1/2)**(1/2),
                        'ms_layer2_init': (1/6)**(1/2),
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