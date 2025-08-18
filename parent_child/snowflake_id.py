import threading
import time


class SnowflakeGenerator:
    """Simple Snowflake-style 64-bit ID generator.

    Layout (64-bit signed int):
    - 41 bits: milliseconds since custom epoch
    - 10 bits: worker id (0-1023)
    - 12 bits: sequence within same ms (0-4095)
    """

    EPOCH = 1704067200000  # Jan 1, 2024 UTC in ms

    def __init__(self, worker_id: int = 1):
        if not (0 <= worker_id < 1024):
            raise ValueError("worker_id must be in [0, 1023]")
        self.worker_id = worker_id
        self._last_ms = -1
        self._sequence = 0
        self._lock = threading.Lock()

    def _current_ms(self) -> int:
        return int(time.time() * 1000)

    def next_id(self) -> int:
        with self._lock:
            now = self._current_ms()
            if now == self._last_ms:
                self._sequence = (self._sequence + 1) & 0xFFF  # 12 bits
                if self._sequence == 0:
                    # sequence overflow in same ms; spin to next ms
                    while now <= self._last_ms:
                        now = self._current_ms()
            else:
                self._sequence = 0

            self._last_ms = now

            ts = (now - self.EPOCH) & ((1 << 41) - 1)
            wid = self.worker_id & ((1 << 10) - 1)
            seq = self._sequence & ((1 << 12) - 1)

            snowflake = (ts << (10 + 12)) | (wid << 12) | seq
            # ensure signed 64-bit range
            if snowflake >= (1 << 63):
                snowflake -= (1 << 64)
            return snowflake
