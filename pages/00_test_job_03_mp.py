import time
import streamlit as st

def main():
    from lib.backgroundjob.process_worker_store import Worker, ProcessWorkerStoreSingleton
    from lib.backgroundjob.test_routine import test_routine
    st.write(ProcessWorkerStoreSingleton.d_workers)
    worker: Worker = ProcessWorkerStoreSingleton.get("my-name")
    if worker:
        st.write(worker.status.value, worker.is_alive())

    with st.sidebar:
        is_running = worker is not None and worker.is_alive()
        is_ready = not is_running
        if st.button('Start worker', disabled=is_running):
            worker = Worker(routine = test_routine, daemon=True)
            ProcessWorkerStoreSingleton.set("my-name", worker)
            worker.start()
            st.experimental_rerun()
            
        if st.button('Stop worker', disabled=is_ready):
            worker.should_stop.set()
            worker.join()
            # worker = None
            ProcessWorkerStoreSingleton.set("my-name", worker)
            st.experimental_rerun()

    if worker is None:
        st.markdown('No worker running.')
    else:
        st.markdown(f'worker: {worker.pid}')
        placeholder = st.empty()
        while worker.is_alive():
            placeholder_container = placeholder.container()
            placeholder_container.markdown(f'status: {worker.status.value}')
            placeholder_container.markdown(f'updated_at: {worker.updated_at.value}')
            placeholder_container.markdown(f'message: {worker.message.value}')
            placeholder_container.markdown(f'tracebacks: {worker.tracebacks[:]}')
            time.sleep(1)
    st.info("worker is dead")
    # if worker:
    #     st.markdown(f'status: {worker.status.value}')
    #     st.markdown(f'updated_at: {worker.updated_at.value}')
    #     st.markdown(f'message: {worker.message.value}')
    #     st.markdown(f'tracebacks: {worker.tracebacks[:]}')

if __name__ == '__main__':
    main()
