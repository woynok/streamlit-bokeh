import multiprocessing
import traceback
from ctypes import c_wchar_p
from datetime import datetime
from pytz import timezone
from lib.backgroundjob.test_routine import test_routine
tz = timezone('Asia/Tokyo')

class Worker(multiprocessing.Process):
    def __init__(self, routine, routine_args = None, routine_kwargs = None, **kwargs):
        super().__init__(**kwargs)
        self.routine = test_routine
        # self.routine = routine
        # self.routine_args = routine_args
        # self.routine_kwargs = routine_kwargs
        if routine_args is None:
            routine_args = []
        if routine_kwargs is None:
            routine_kwargs = {}
        self.routine_args = routine_args
        self.routine_kwargs = routine_kwargs
        time_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        self.should_stop = multiprocessing.Event()
        self.status = multiprocessing.Value(c_wchar_p, "ready")
        self.updated_at = multiprocessing.Value(c_wchar_p, time_str)
        self.message = multiprocessing.Value(c_wchar_p, "")
        self.tracebacks = multiprocessing.Array(c_wchar_p, 100)
        
    def run(self):
        self.status = "running"
        self.updated_at = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.routine(*self.routine_args, **self.routine_kwargs)
            self.status = "done"
            self.updated_at = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            print("print done from worker")
        except Exception as e:
            self.updated_at = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            self.status = f"error at app"
            self.message = str(e)
            self.tracebacks = traceback.format_exc().splitlines()
            print(e)
            print(traceback.format_exc())
            print("print error from worker")

class ProcessWorkerStoreSingleton:
    d_workers = {}

    @classmethod
    def get(cls, key):
        return cls.d_workers.get(key)
    
    @classmethod
    def set(cls, key, worker):
        cls.d_workers[key] = worker

    def is_running(self, key):
        worker = self.get(key)
        return worker is not None and worker.is_alive()