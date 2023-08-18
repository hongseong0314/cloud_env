import pickle
import torch
import time
import numpy as np
import random
from tqdm import tqdm
import os
# base
from codes.job import Job
from codes.machine import MachineConfig

# env
from independent_job.env import Env

# model
from independent_job.matrix.alg import MatrixAlgorithm
from independent_job.matrix.model import BGC
from independent_job.config import matrix_config

from independent_job.fit.alg import FitAlgorithm
from independent_job.fit.model import Fit
from independent_job.config import fit_config

# log
import wandb

class trainer():
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.model_name == 'matrix':
            self.agent = BGC(cfg)
            self.algorithm = lambda agent : MatrixAlgorithm(agent)
            self.name = f"{self.cfg.model_name}-{cfg.model_params['TMHA']}-{cfg.model_params['MMHA']}"
            
        elif cfg.model_name == 'fit':
            self.agent = Fit(cfg)
            self.algorithm = lambda cfg : FitAlgorithm(cfg)
            self.name = f"{self.cfg.model_name}"
            

        # cpu, mem, pf, ps, pe
        self.m_resource_config = [[32, 80, 120, 80, 40], 
                                    [1, 2, 3, 1.5, 1], 
                                    [675.7838, 643.8629,258.2628,332.1814,119.0417], 
                                    [193.4651, 193.8555,66.8607,101.3687,45.3834], 
                                    [0.9569,0.7257,1.5767,0.7119,1.5324]]

        with open('./train.pkl', 'rb') as f:
            self.train_task = pickle.load(f)
        with open('./valid.pkl', 'rb') as f:
            self.valid_task = pickle.load(f)

    def setup(self):
        cpus, mems, pfs, pss, pes = self.m_resource_config
        self.cfg.cpus_max = max(cpus)
        self.cfg.mems_max = max(mems)
        self.cfg.machine_configs = [MachineConfig(cpu_capacity=cpu, 
                                 memory_capacity=mem_disk, 
                                 disk_capacity=mem_disk,
                                 pf=pf, ps=ps, pe=pe) for cpu, mem_disk, pf, ps, pe in zip(cpus,\
                mems, pfs, pss, pes)]

        wandb.init(project='cloud')
        wandb.run.name = self.name
        wandb.run.save()

        wandb.config.update(self.cfg.to_dict())
        
    def fit(self):
        self.setup()

        with tqdm(range(self.cfg.epoch), unit="Run") as runing_bar:
            for i in runing_bar:
                self.agent.scheduler.step()
                loss, clock, energy = self.training()
                valid_clock, valid_energy = self.valiing()

                runing_bar.set_postfix(loss=loss,
                                   valid_clock=valid_clock,
                                   valid_energy=valid_energy,)

                self.cfg.file.write(f"loss : {loss}, clock : {valid_clock}, energy : {valid_energy}\n")

                wandb.log({"Training loss": loss,
                           "Training clock": clock,
                           "Training energy": energy,
                           "valid_clock": valid_clock,
                           "valid_energy": valid_energy})
                
            
    def roll_out(self):
        clock_list, energy_list = [], []

        for _ in range(6):
            random.shuffle(cfg.machine_configs)
            algorithm = self.algorithm(self.agent)
            sim = Env(self.cfg)
            sim.setup()
            sim.episode(algorithm)
            eg = sim.total_energy_consumptipn
            self.agent.trajectory(-eg)
            clock_list.append(sim.time)
            energy_list.append(eg)

        loss = self.agent.update_parameters()
        return loss, np.mean(clock_list), np.mean(energy_list)
    
    def training(self):
        losses, clocks, energys = [], [], []
        self.agent.model.train()
        for i in range(self.cfg.job_len, len(self.train_task))[:self.cfg.train_len]:
            torch.cuda.empty_cache()
            self.cfg.task_configs = self.train_task[i-self.cfg.job_len:i]
            loss, clock, energy = self.roll_out()
            losses.append(loss)
            clocks.append(clock)
            energys.append(energy)
        return np.mean(losses), np.mean(clocks), np.mean(energys)

    def valiing(self):
        clocks, energys = [], []
        self.agent.model.eval()
        for i in range(self.cfg.job_len, len(self.valid_task))[:self.cfg.valid_len]:
            self.cfg.task_configs = self.valid_task[i-self.cfg.job_len:i]

            algorithm = self.algorithm(self.agent)
            sim = Env(self.cfg)
            sim.setup()
            sim.episode(algorithm)
            eg = sim.total_energy_consumptipn
            clocks.append(sim.time)
            energys.append(eg)

        return np.mean(clocks), np.mean(energys)
    
if __name__ == '__main__':
    # seed
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED) 
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # epoch
    epoch = 100
    
    # job len
    train_len = 10
    valid_len = 5
    job_len = 3

    model_name = 'matrix'
    
    if model_name == 'matrix':
        # base parm
        cfg = matrix_config()
        cfg.model_name = model_name
        cfg.model_params['skip'] = False

        # epoch
        cfg.epoch = epoch
        
        # job len
        cfg.train_len = train_len
        cfg.valid_len = valid_len
        cfg.job_len = job_len

        cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        cfg.model_params['device'] = cfg.device

        # encoder type
        cfg.model_params['TMHA'] = 'depth'
        cfg.model_params['MMHA'] = 'depth'

        # model_name/epoch/train_len/valid_len/job_len/TMHA/MMHA/seed
        cfg.model_params['save_path'] = '{}_{}_{}_{}_{}_{}_{}_{}_skip.pth'.format(
                                                                cfg.model_name,
                                                                cfg.epoch,
                                                                cfg.train_len,
                                                                cfg.valid_len,
                                                                cfg.job_len,
                                                                cfg.model_params['TMHA'],
                                                                cfg.model_params['MMHA'],
                                                                SEED)
        
    elif model_name == 'fit':
        cfg = fit_config()
        cfg.model_name = model_name
        
        # epoch
        cfg.epoch = epoch
        
        # job len
        cfg.train_len = train_len
        cfg.valid_len = valid_len
        cfg.job_len = job_len

        cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        cfg.model_params['device'] = cfg.device

        # model_name/epoch/train_len/valid_len/job_len//seed
        cfg.model_params['save_path'] = '{}_{}_{}_{}_{}_{}.pth'.format(
                                                                cfg.model_name,
                                                                cfg.epoch,
                                                                cfg.train_len,
                                                                cfg.valid_len,
                                                                cfg.job_len,
                                                                SEED)

    # model_name
    file_name = "depth.txt"
    with open(file_name, "a") as file:
        cfg.file = file
        st = time.time()
        triner = trainer(cfg)
        triner.fit()
        print(time.time() - st)
    
