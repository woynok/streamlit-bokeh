from __future__ import annotations
from typing import Literal
import os
import yaml
import traceback
from dataclasses import dataclass
from datetime import datetime
from pytz import timezone
from ctypes import c_wchar_p
import multiprocessing

from lib.backgroundjob.test_routine import test_routine

tz = timezone('Asia/Tokyo')

@dataclass
class WorkerInfo:
    name: str
    _last_info_updated_at: str = ""
    last_status: Literal["not initiated", "ready", "running", "done", "error at app", "error at worker"] = "not initiated"
    created_at: str = ""
    updated_at: str = ""
    message: str = ""
    tracebacks: list = None

    @property
    def last_info_updated_at(self)->str:
        return self._last_info_updated_at

    def to_dict(self)->dict:
        return {
            "name": self.name,
            "last_status": self.last_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message": self.message,
            "tracebacks": [t for t in self.tracebacks if t is not None]
        }
    
    @classmethod
    def _from_dict(cls, d)->WorkerInfo:
        return cls(**d)

    def save(self):
        filepath = f"background_job/info_{self.name}.yaml"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f)
    
    @classmethod
    def _from_file(cls, name)->WorkerInfo:
        import yaml
        filepath = f"background_job/info_{name}.yaml"
        with open(filepath, "r") as f:
            return cls._from_dict(yaml.load(f, Loader=yaml.FullLoader))

    def is_old_info(self, ttl = 0.5)->bool:
        try:
            last_info_updated_at = datetime.strptime(self._last_info_updated_at, "%Y-%m-%d %H:%M:%S")
            return (datetime.now(tz) - last_info_updated_at).total_seconds() > ttl
        except TypeError:
            return True
        except ValueError:
            return True
    
    def is_running_too_long(self, ttl = 300)->bool:
        updated_at = datetime.strptime(self.updated_at, "%Y-%m-%d %H:%M:%S")
        # offset aware updated_at by timezone
        updated_at = updated_at.astimezone(tz)
        return (datetime.now(tz) - updated_at).total_seconds() > ttl
        
    @classmethod
    def get_from_file(cls, name)->WorkerInfo:
        try:
            out = cls._from_file(name)
        except FileNotFoundError:
            created_at = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            out = cls(name = name, created_at = created_at)
        out._last_info_updated_at = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return out
    
    @classmethod
    def get_from_worker(cls, worker: Worker, save = True)->WorkerInfo:
        out = cls(name = worker.name).update_values_by_worker(worker)
        if save:
            out.save()
        return out

    def update_values_by_worker(self, worker: Worker, save = True)->WorkerInfo:
        self._last_info_updated_at = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        self.last_status = worker._status.value
        self.updated_at = worker._updated_at.value
        self.message = worker._message.value
        self.tracebacks = worker._tracebacks[:]
        if save:
            self.save()
        return self
    
    @property
    def status(self)->str:
        if self.is_running_too_long():
            return "error at worker"
        else:
            return self.last_status
    
    __repr__ = __str__ = lambda self: f"<WorkerInfo name={self.name} status={self.status} created_at={self.created_at} updated_at={self.updated_at} message={self.message} tracebacks={self.tracebacks}>"
    
class Worker(multiprocessing.Process):
    def __init__(self, name, routine, routine_args = None, routine_kwargs = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.routine = routine
        if routine_args is None:
            routine_args = []
        if routine_kwargs is None:
            routine_kwargs = {}
        self.routine_args = routine_args
        self.routine_kwargs = routine_kwargs
        time_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        self.should_stop = multiprocessing.Event()
        self._status = multiprocessing.Value(c_wchar_p, "ready")
        self._updated_at = multiprocessing.Value(c_wchar_p, time_str)
        self._message = multiprocessing.Value(c_wchar_p, "")
        self._tracebacks = multiprocessing.Array(c_wchar_p, 100)
        self._info = WorkerInfo.get_from_worker(self)
    
    @property
    def info(self)->WorkerInfo:
        return WorkerInfo.get_from_file(self.name)

    def update_info_values(self, last_status = "", message = "", tracebacks = None):
        if tracebacks is None:
            tracebacks = []
        updated_at = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        self._status = multiprocessing.Value(c_wchar_p, last_status)
        self._message = multiprocessing.Value(c_wchar_p, message)
        self._tracebacks = multiprocessing.Array(c_wchar_p, tracebacks)
        self._updated_at = multiprocessing.Value(c_wchar_p, updated_at)
        self._info.update_values_by_worker(self)

    def run(self):
        self.update_info_values(
            last_status = "running"
        )
        print("print running from worker")
        print(self._updated_at.value)
        try:
            self.routine(*self.routine_args, **self.routine_kwargs)
            self.update_info_values(
                last_status = "done"
            )
            print("print done from worker")
            print(self._updated_at.value)
        except Exception as e:
            self.update_info_values(
                last_status = "error at app",
                message = str(e),
                tracebacks = traceback.format_exc().splitlines()
            )
            print(self._updated_at.value)
            print(e)
            print(traceback.format_exc())
            print("print error from worker")

class ProcessWorkerStoreSingleton:
    d_workers = {}

    @classmethod
    def get(cls, worker_name):
        return cls.d_workers.get(worker_name)
    
    @classmethod
    def set(cls, worker):
        cls.d_workers[worker.name] = worker

    @classmethod
    def is_running(cls, worker_name):
        worker = cls.get(worker_name)
        if worker is not None:
            return worker.is_alive()
        else:
            return False
    
    @classmethod
    def get_n_running(cls)->int:
        return sum([cls.is_running(worker_name) for worker_name in cls.d_workers.keys()])

    @classmethod
    def is_ok_to_start(cls, max_n_workers = 2)->bool:
        return cls.get_n_running() < max_n_workers
    
    @classmethod
    def get_worker_info(cls, worker_name)->WorkerInfo:
        worker = cls.get(worker_name)
        if worker is not None:
            return worker.info
        else:
            return None
    
    @classmethod
    def stop(cls, worker_name)->bool:
        worker: Worker = cls.get(worker_name)
        if worker is not None:
            worker.should_stop.set()
            worker.join()
            worker.terminate()
            cls.d_workers.pop(worker_name)
            return True
        else:
            return False
    
    @classmethod
    def get_all_worker_info_list(cls)->list[WorkerInfo]:
        info_list = []
        for worker_name in cls.d_workers.keys():
            d = cls.get_worker_info(worker_name).to_dict()
            d["is_alive"] = cls.is_running(worker_name)
            info_list.append(d)
        return info_list

    @classmethod
    def start_worker(cls, worker_name, routine, routine_args = None, routine_kwargs = None)->Worker:
        if routine_args is None:
            routine_args = []
        if routine_kwargs is None:
            routine_kwargs = {}
        worker = Worker(name = worker_name, routine = routine, routine_args = routine_args, routine_kwargs = routine_kwargs, daemon=True)
        cls.set(worker)
        worker.start()
        return worker
    