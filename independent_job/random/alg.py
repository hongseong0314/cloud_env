import torch
class RandomAlgorithm:
    def __init__(self):
        pass
    def __call__(self, state):
        ninf_mask = state.ninf_mask
        available_task_idx = state.available_task_idx

        m_idx, t_idx = torch.nonzero((~torch.isinf(ninf_mask)).squeeze(0), as_tuple=True)

        pair_index = torch.randint(len(m_idx),(1,))

        machine_pointer = m_idx[pair_index].item()
        task_pointer = available_task_idx[int(t_idx[pair_index])].item()

        return machine_pointer, task_pointer