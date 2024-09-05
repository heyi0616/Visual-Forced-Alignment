import math


class OriginalOpt:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._step = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def set_step(self, step):
        self._step = step

    def get_step(self):
        return self._step

    def step(self):
        "Update parameters and rate"
        self._step += 1
        self.optimizer.step()

    def rate(self):
        return self.optimizer.param_groups[0]['lr']

    def get_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


class NoamOpt(OriginalOpt):
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer, max_step=90000):
        """Construct an NoamOpt object."""
        super(NoamOpt, self).__init__(optimizer)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.max_step = max_step
        self.peak_lr = self.factor * self.model_size ** (-0.5) / math.sqrt(warmup)

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        if self._step <= self.max_step:  # 超过之后恒定学习率然后利用scheduler衰减
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return (
            self.factor
            * self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }
