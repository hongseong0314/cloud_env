import pickle
import torch
# base
from codes.job import Job
from codes.machine import MachineConfig

# env
from independent_job.env import Env

# model
from independent_job.matrix.alg import MatrixAlgorithm
from independent_job.matrix.model import BGC
from independent_job.config import matrix_config

##########################
cfg = matrix_config()

# machine
m_resource_config = [[32, 32, 32, 32, 32],
                                    [1, 1, 1, 1, 1],
                                    [675.7838, 643.8629,258.2628,332.1814,119.0417],
                                    [193.4651, 193.8555,66.8607,101.3687,45.3834],
                                    [0.9569,0.7257,1.5767,0.7119,1.5324]]
cpus, mems, pfs, pss, pes = m_resource_config
cfg.cpus_max = max(cpus)
cfg.mems_max = max(mems)
cfg.machine_configs = [MachineConfig(cpu_capacity=cpu,
                        memory_capacity=mem_disk,
                        disk_capacity=mem_disk,
                        pf=pf, ps=ps, pe=pe) for cpu, mem_disk, pf, ps, pe in zip(cpus,\
    mems, pfs, pss, pes)]

# task
with open('./train.pkl', 'rb') as f:
    train_task = pickle.load(f)

task_configs =  train_task[:3]
cfg.task_configs = task_configs

# model
cfg.model_name = 'matrix'
cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
cfg.model_params['device'] = cfg.device
cfg.model_params['TMHA'] = 'depth'
cfg.model_params['MMHA'] = 'depth'
cfg.model_params['save_path'] = 'test.pth'
cfg.agent = BGC(cfg)

file_name = "data.txt"
with open(file_name, "a") as file:
    cfg.file = file
    alg = MatrixAlgorithm(cfg)
    env = Env(cfg)
    env.setup()
    env.episode(alg)
