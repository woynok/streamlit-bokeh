import time
import multiprocessing
import streamlit as st
from lib.backgroundjob.process_worker_store import ProcessWorkerStoreSingleton

class Worker(multiprocessing.Process):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_stop = multiprocessing.Event()
        self.counter = multiprocessing.Value('i', 0)
        
    def run(self):
        while not self.should_stop.wait(0):
            time.sleep(1)
            self.counter.value += 1

def main():
    st.write(ProcessWorkerStoreSingleton.d_workers)
    worker: Worker = ProcessWorkerStoreSingleton.get("my-name")

    with st.sidebar:
        if st.button('Start worker', disabled=worker is not None):
            worker = Worker(daemon=True)
            ProcessWorkerStoreSingleton.set("my-name", worker)
            worker.start()
            st.experimental_rerun()
            
        if st.button('Stop worker', disabled=worker is None):
            worker.should_stop.set()
            worker.join()
            worker = None
            ProcessWorkerStoreSingleton.set("my-name", worker)
            st.experimental_rerun()

    if worker is None:
        st.markdown('No worker running.')
    else:
        st.markdown(f'worker: {worker.pid}')
        placeholder = st.empty()
        while worker.is_alive():
            placeholder.markdown(f'counter: {worker.counter.value}')
            time.sleep(1)

if __name__ == '__main__':
    main()
