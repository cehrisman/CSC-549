import collections
import numpy as np

Experience = collections.namedtuple("Experience", field_names=["state", "action", "reward",  "next_state", "done"])


class ExperienceReplay:
    def __init__(self, batch_size, buffer_size=None, random_state=None):
        self.batch_size_ = batch_size
        self.buffer_size_ = buffer_size
        self.buffer_ = collections.deque(maxlen=buffer_size)
        self.random_state_ = np.random.RandomState() if random_state is None else random_state

    def batch_length(self):
        return len(self.buffer_)

    def batch_size(self):
        return self.batch_size_

    def buffer_size(self):
        return self.buffer_size_

    def append(self, experience):
        self.buffer_.append(experience)

    def sample(self):
        idxs = self.random_state_.randint(len(self.buffer_), size=self.batch_size_)
        experiences = [self.buffer_[idx] for idx in idxs]
        return experiences
