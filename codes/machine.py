import torch

class MachineConfig(object):
    def __init__(self, cpu_capacity, memory_capacity, disk_capacity,
                 cpu=None, memory=None, disk=None,
                 pf=None, ps=None, pe=None, mips=None):
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.disk_capacity = disk_capacity

        self.cpu = cpu_capacity if cpu is None else cpu
        self.memory = memory_capacity if memory is None else memory
        self.disk = disk_capacity if disk is None else disk
        self.mips = 0 if mips is None else mips
        self.pf = 0 if pf is None else pf
        self.ps = 0 if ps is None else ps
        self.pe = 0 if pe is None else pe

class Machine:
    def __init__(self, machine_config):
        self.cpu_capacity = machine_config.cpu_capacity
        self.memory_capacity = machine_config.memory_capacity
        self.disk_capacity = machine_config.disk_capacity
        self.cpu = machine_config.cpu
        self.memory = machine_config.memory
        self.disk = machine_config.disk
        self.mips = machine_config.mips
        self.pf = machine_config.pf
        self.ps = machine_config.ps
        self.pe = machine_config.pe
        self.pre_time = 0
        self.pre_use = 0
        # self.energy_consumption = 0

        # machine state
        self.init_state = torch.tensor([self.cpu_capacity, self.memory_capacity])
        self.state = torch.tensor([self.cpu, self.memory])

        # [start, end, cpu, mem]
        self.running_tasks = torch.zeros(size=(0, 4))
        self.finished_tasks = torch.zeros(size=(0, 4))

    def finished_task_move(self, time):
        run_idx = (self.running_tasks[..., 0] <=  time) & (time <= self.running_tasks[..., 1])

        self.finished_tasks = torch.cat((self.finished_tasks, self.running_tasks[~run_idx]), dim=0)
        self.running_tasks = self.running_tasks[run_idx]

        self.state = self.init_state - self.running_tasks[:, [2, 3]].sum(dim=0)

    def allocation(self, task):
        self.running_tasks = torch.cat((self.running_tasks, task), dim=0)

    def energy_consumption(self):
        use_cpu = (self.cpu_capacity - self.state[0].item()) / self.cpu_capacity
        return (self.ps + (self.pf - self.ps) * (use_cpu ** self.pe))
    
    def running(self):
        return self.state[0].item() != self.cpu_capacity