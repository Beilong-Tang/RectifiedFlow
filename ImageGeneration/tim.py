from contextlib import contextmanager
import time
from datetime import timedelta
from contextlib import contextmanager

def time_to_str(time):
    return str(timedelta(seconds=int(time)))

class ContextTimer:
    """Flexible timer with context manager support for multiple processes."""
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.step_count = 0
        self.timers = {}                # accumulated time per key
        self.total_step_time = 0.0      # total elapsed time for all steps
        self.step_start_time = None     # start time of current step

    @contextmanager
    def track(self, key):
        """Use like `with timer.track("data"):`"""
        t0 = time.time()
        yield
        elapsed = time.time() - t0
        self.timers[key] = self.timers.get(key, 0.0) + elapsed

    def step_start(self):
        """Call at the beginning of each step to mark the start."""
        self.step_start_time = time.time()

    def step_end(self):
        """Call at the end of the step."""
        if self.step_start_time is None:
            raise RuntimeError("step_start() must be called before step_end()")
        elapsed = time.time() - self.step_start_time
        self.total_step_time += elapsed
        self.step_count += 1
        self.step_start_time = None   # reset for next step

    def stats(self):
        if self.step_count == 0:
            step_times = {}
            step_total = 0.0
        else:
            step_times = {k: t / self.step_count for k, t in self.timers.items()}
            step_total = self.total_step_time / self.step_count

        # Build string components
        eta = timedelta(seconds=int((self.total_steps - self.step_count) * step_total)) if step_total > 0 else timedelta(0)
        eta = str(eta)
        res = f"[{time_to_str(self.total_step_time)}<{eta}, {step_total:.3f}s/it"
        for k, v in step_times.items():
            res += f", {k}: {v:.3f}"
        res +="]"
        return res

if __name__ == "__main__":
    total_steps = 100
    tim = ContextTimer(total_steps=100)

    for i in range(0, 100):
        tim.step_start()
        time.sleep(0.1)
        print(tim.stats())
        tim.step_end()
        pass
