from time import perf_counter


class Timer:

    def __init__(self, tag: str | None = None):
        self._tag = tag
        self._start = None
        self._stop = None

    def start(self):
        self._start = perf_counter()
        return self

    def result(self):
        return self._stop - self._start

    def stop(self, print_result: bool = True):
        self._stop = perf_counter()
        if print_result: self.print_result()

    def print_result(self):
        if self._tag:
            print(f"Result ({self._tag}): {self.result()}")
            return
        print(f"Result: {self.result()}")
