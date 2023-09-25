from __future__ import annotations
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, Future

import time
from datetime import datetime
from pytz import timezone
timezone_tokyo = timezone('Asia/Tokyo')

class HeartBeat:
    def __init__(self, pulse: Future, future: Future):
        self.pulse: Future = pulse
        self.future: Future = future
    
    def __enter__(self)->HeartBeat:
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.pulse.cancel()
    
    def get_result(self):
        return self.future.result()
    
    @property
    def result(self):
        return self.get_result()

class Heart(ThreadPoolExecutor):
    """while Heart is swinging, print the time every pulse_duration seconds
    """
    def __init__(
            self,
            routine: Callable,
            pulse_duration: int = 30,
            pulse_function: Callable = None,
            ):

        self.pulse_duration: int = pulse_duration
        if pulse_function is None:
            self.pulse_function: Callable = self._pulse_function_default
        else:
            self.pulse_function: Callable = pulse_function
        self.routine = routine
        self.pulse: Future = None
        self.future: Future = None
        super().__init__(max_workers=2)
    
    def _pulse_function_default(self):
        print(f"{datetime.now(tz=timezone_tokyo)}::send pulse")
    
    def beat(self, *routine_args, **routine_kwargs)->HeartBeat:
        self.pulse = self.submit(self._send_pulse)
        self.future = self.submit(self.routine, *routine_args, **routine_kwargs)
        pulse_check = self.pulse.result()
        return HeartBeat(
            pulse = self.pulse,
            future = self.future,
            )
    
    def _send_pulse(self):
        while self.future is None or self.future.done() == False:
            self.pulse_function()
            time.sleep(self.pulse_duration)


if __name__ == '__main__':
    import streamlit as st
    def very_slow_routine(coefficient = 1.0e-4):
        print(f"{datetime.now(tz=timezone_tokyo)}::very_slow_routine() starting...")
        for i in range(10000):
            for j in range(10000):
                c = i * j
        print(f"{datetime.now(tz=timezone_tokyo)}::very_slow_routine() done...")
        return c * coefficient

    placeholder_button = st.empty()
    button = placeholder_button.button("start", key="start_initial")
    if button:
        placeholder_button.button("start", disabled=True, key="start_disabled")
        with st.spinner():
            heart = Heart(pulse_duration = 1.0, routine = very_slow_routine)
            with heart.beat(coefficient = 1.0e-8) as heart_beat:
                result = heart_beat.get_result()
        st.write(result)
        placeholder_button.button("start", key="start_enabled")
