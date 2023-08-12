from dataclasses import dataclass

@dataclass
class Job:
    submit_time: int = None
    tasks : list = None