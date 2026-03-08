from time import perf_counter


class Timer:

    def __init__(self):
        self._start = None
        self._stop = None

    def start(self):
        self._start = perf_counter()

    def result(self):
        return self._stop - self._start

    def stop(self, print_result: bool = True):
        self._stop = perf_counter()
        if print_result: self.print_result()

    def print_result(self):
        print(f"Result: {self.result()}")
