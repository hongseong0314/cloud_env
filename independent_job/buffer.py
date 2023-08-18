import numpy as np

class trajectory_buffer():
    def __init__(self, 
                 buffer_limit=10000, 
                 batch_size=64):
        self.ss_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)

        self.buffer_limit = buffer_limit
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def put(self, sample):
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.buffer_limit

        self.size += 1
        self.size = min(self.size, self.buffer_limit)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)

        s_lst = np.vstack(self.ss_mem[idxs])
        a_lst = np.vstack(self.as_mem[idxs])
        r_lst = np.vstack(self.rs_mem[idxs])  
        s_prime_lst = np.vstack(self.ps_mem[idxs])
        d_lst = np.vstack(self.ds_mem[idxs])

        experiences = s_lst, \
                      a_lst, \
                      r_lst, \
                      s_prime_lst, \
                      d_lst, \
                          
        return experiences

    def __len__(self):
        return self.size