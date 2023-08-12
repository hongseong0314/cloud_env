class MatrixAlgorithm:
    def __init__(self, cfg):
        self.device = cfg.device
        self.agent = cfg.agent
        self.current_trajectory = []
        self.file = cfg.file

    def __call__(self, state):
        machine_feature = state.machine_feature.to(self.device)
        task_feature = state.task_feature.to(self.device)
        D_TM = state.D_TM.to(self.device)
        ninf_mask = state.ninf_mask.to(self.device)
        available_task_idx = state.available_task_idx
        task_num = task_feature.size(1)

        task_selected = self.agent.decision(machine_feature, task_feature, D_TM, ninf_mask)
        machine_pointer = task_selected // task_num
        task_pointer = task_selected % task_num

        # self.file.write(f"selected : {task_selected}\n")
        return machine_pointer, available_task_idx[int(task_pointer)].item()