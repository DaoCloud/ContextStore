from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import torch


class AsyncIOManager:
    def __init__(self, num_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._layer_events: dict[str, threading.Event] = {}
        self._save_futures: list[Future] = []
        self._lock = threading.Lock()

    def submit_load_layer(
        self,
        layer_name: str,
        load_fn: Callable[[], None],
    ) -> None:
        event = threading.Event()
        with self._lock:
            self._layer_events[layer_name] = event

        def _do_load():
            try:
                load_fn()
            finally:
                event.set()

        self._executor.submit(_do_load)

    def wait_for_layer(self, layer_name: str) -> None:
        with self._lock:
            event = self._layer_events.get(layer_name)
        if event is not None:
            event.wait()

    def submit_save(self, save_fn: Callable[[], None]) -> None:
        future = self._executor.submit(save_fn)
        with self._lock:
            self._save_futures.append(future)

    def wait_all_saves(self) -> None:
        with self._lock:
            futures = list(self._save_futures)
            self._save_futures.clear()
        for f in futures:
            f.result()

    def reset(self) -> None:
        with self._lock:
            self._layer_events.clear()
            self._save_futures.clear()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
