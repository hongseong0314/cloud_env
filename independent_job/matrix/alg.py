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

        if self.agent.skip:
            task_selected = self.agent.decision(machine_feature, task_feature, D_TM, ninf_mask)
            machine_pointer = task_selected // (task_num+1)
            task_pointer = task_selected % (task_num+1)
            if task_pointer == 0:
                return None, None
            else:
                return machine_pointer, available_task_idx[int(task_pointer)-1].item()

        else: 
            task_selected = self.agent.decision(machine_feature, task_feature, D_TM, ninf_mask)
            machine_pointer = task_selected // task_num
            task_pointer = task_selected % task_num

            return machine_pointer, available_task_idx[int(task_pointer)].item()

# import torch
# class MatrixAlgorithm:
#     def __init__(self, agent):
#         self.agent = agent
#         self.device = agent.device

#         if self.agent.model.training:
#             self.logpa_list = torch.zeros(size=(1, 0)).to(self.device)
        
#     def __call__(self, state):
#         machine_feature = state.machine_feature
#         task_feature = state.task_feature
#         D_TM = state.D_TM
#         ninf_mask = state.ninf_mask
#         available_task_idx = state.available_task_idx
#         task_num = task_feature.size(1)

#         if self.agent.skip:
#             task_selected,logpa = self.agent.decision(machine_feature, task_feature, D_TM, ninf_mask)
#             machine_pointer = task_selected // (task_num+1)
#             task_pointer = task_selected % (task_num+1)

#             if self.agent.model.training:
#                 self.logpa_list = torch.cat((self.logpa_list, logpa[None, ...]), dim=-1)
            
#             if task_pointer == 0:
#                 return None, None
#             else:
#                 return machine_pointer, available_task_idx[int(task_pointer)-1].item()

#         else: 
#             task_selected,logpa = self.agent.decision(machine_feature, task_feature, D_TM, ninf_mask)
#             machine_pointer = task_selected // task_num
#             task_pointer = task_selected % task_num

#             if self.agent.model.training:
#                 self.logpa_list = torch.cat((self.logpa_list, logpa[None, ...]), dim=-1)
            
#             return machine_pointer, available_task_idx[int(task_pointer)].item()