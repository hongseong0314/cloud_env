class MatrixAlgorithm:
    def __init__(self, agent):
        self.agent = agent
        
    def __call__(self, state):
        machine_feature = state.machine_feature
        task_feature = state.task_feature
        D_TM = state.D_TM
        ninf_mask = state.ninf_mask
        available_task_idx = state.available_task_idx
        task_num = task_feature.size(1)

        task_selected = self.agent.decision(machine_feature, task_feature, D_TM, ninf_mask)
        machine_pointer = task_selected // task_num
        task_pointer = task_selected % task_num

        return machine_pointer, available_task_idx[int(task_pointer)].item()