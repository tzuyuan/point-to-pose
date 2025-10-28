# workers.py (module-level so it's picklable)
import multiprocessing as mp
from queue import Empty
import time

import point2pose.modules as _modules  # trigger registrations
from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import OPTIMIZER
from point2pose.modules.optimizer.isam2_optimizer import ISAM2Optimizer


def optimizer_worker(cfg, in_q: mp.Queue, out_q: mp.Queue, stop_evt: mp.Event):
    # Build optimizer INSIDE child process

    # optimizer = build_from_cfg(cfg, OPTIMIZER)  # pure-Python cfg only
    optimizer = ISAM2Optimizer(cfg)

    while not stop_evt.is_set():
        try:
            pkt = in_q.get_nowait()  # your compact packet (np arrays ok)
            # TODO: do we need this for resource competition?
            time.sleep(0.00001)
        except Empty:
            continue

        # One incremental step; returns a small, picklable result
        result = optimizer.optimize(pkt)
        # latest-wins: keep queue small
        if out_q.full():
            try:
                out_q.get_nowait()
            except Empty:
                pass
        out_q.put_nowait(result)
