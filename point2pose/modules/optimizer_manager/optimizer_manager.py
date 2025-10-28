import multiprocessing as mp

from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import OPTIMIZER
from point2pose.modules.object.object import Object
from point2pose.data_types.optimizer_result import OptimizerResult
from point2pose.modules.optimizer_manager.optimizer_worker import optimizer_worker


class OptimizerManager:
    def __init__(self, config):
        self.config = config
        self.num_obj = 0
        self.optimizers = []
        self.workers = {}  # {obj_id -> (process, input_queue, output_queue)}

        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            # Already set elsewhere — safe to ignore
            pass

    def initialize(self, num_obj):
        self.num_obj = num_obj
        for obj_id in range(num_obj):
            input_queue = mp.Queue(maxsize=2)
            output_queue = mp.Queue(maxsize=2)
            stop_event = mp.Event()
            self.workers[obj_id] = (
                mp.Process(
                    target=optimizer_worker,
                    args=(self.config, input_queue, output_queue, stop_event),
                ),
                input_queue,
                output_queue,
                stop_event,
            )

    def start(self):
        for obj_id in range(self.num_obj):
            self.workers[obj_id][0].start()

        print(f"[OptimizerManager] Started {self.num_obj} optimizer workers")

    def stop(self):
        for _, (p, _, _, stop_evt) in self.workers.items():
            stop_evt.set()
            p.join(timeout=1.0)

    def set_input(self, obj_id: int, object_input: Object):
        if obj_id not in self.workers:
            raise ValueError(
                f"[OptimizerManager] Object {obj_id} not found in workers!!"
            )
        _, input_queue, _, _ = self.workers[obj_id]

        # if the queue is full, get the oldest item
        if input_queue.full():
            input_queue.get_nowait()

        input_queue.put(object_input)

    def get_output(self, obj_id: int) -> OptimizerResult:
        if obj_id not in self.workers:
            raise ValueError(
                f"[OptimizerManager] Object {obj_id} not found in workers!!"
            )
        output = None
        _, _, output_queue, _ = self.workers[obj_id]

        try:
            output = output_queue.get_nowait()
        except Exception:
            output = None

        return output

    def __del__(self):
        self.stop()
