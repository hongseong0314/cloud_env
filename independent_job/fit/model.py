import numpy as np
import torch
import torch.nn.functional as F

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

class Fit(object):
    def __init__(self,cfg):
        super().__init__()
        self.device = cfg.device
        self.gamma = 0.999
        self.model = Qnet(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **cfg.optimizer_params['scheduler'])
        self.save_path = cfg.model_params['save_path']
        self.load_path = cfg.model_params['load_path']

        if self.load_path:
            print(f"load weight : {self.load_path}")
            self.model.load_state_dict(torch.load(cfg.model_params['load_path'],
                                                  map_location=self.device))
            self.model.eval()

        self.logpa_sum_list = torch.zeros(size=(1, 0)).to(cfg.device)
        self.logpa_list = torch.zeros(size=(1, 0)).to(cfg.device)
        self.reward_list = []

    def trajectory(self, reward):
        self.logpa_sum_list = torch.cat((self.logpa_sum_list, self.logpa_list.sum(dim=-1)[None, ...]), dim=-1)
        self.reward_list.append(reward)

        self.logpa_list = torch.zeros(size=(1, 0)).to(self.device)

    def model_save(self):
        torch.save(self.model.state_dict(), self.save_path)

    def decision(self, feature):
        feature = feature.to(self.device)

        if self.model.training:
            logits = \
                    self.model(feature)
            # [B, M*T]
            dist = torch.distributions.Categorical(logits=logits)
            task_selected = dist.sample()
            # [B,]
            logpa = dist.log_prob(task_selected)
            # [B,]
            self.logpa_list = torch.cat((self.logpa_list, logpa[None, ...]), dim=-1)
            return task_selected.item()
        else:
            with torch.no_grad():
               logits = \
                        self.model(feature)
            task_selected = logits.argmax(dim=1)
            logpa = None
            return task_selected.item()

    def update_parameters(self):
        rewards = torch.tensor(self.reward_list).to(self.device)
        advantage = rewards - rewards.float().mean()
        loss = -advantage * self.logpa_sum_list
        loss_mean = loss.mean()

        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        self.model_save()

        self.reward_list = []
        self.logpa_sum_list = torch.zeros(size=(1, 0)).to(self.device)
        return loss_mean.item()

class Qnet(torch.nn.Module):
    def __init__(self, **model_params):
        super(Qnet, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            torch.nn.Linear(model_params['nT'] + model_params['nM'] + 1, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
        )
        self.FC = torch.nn.Linear(32, 1)
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.FC(x)
        return x.squeeze(-1).unsqueeze(0)