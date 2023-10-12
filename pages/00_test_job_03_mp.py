import time
import streamlit as st

def main():
    from lib.backgroundjob.process_workers import Worker, ProcessWorkerStoreSingleton, WorkerInfo
    from lib.backgroundjob.test_routine import test_routine
    st.write(ProcessWorkerStoreSingleton.d_workers)
    worker_unique_name = "my-name"
    worker: Worker = ProcessWorkerStoreSingleton.get(worker_unique_name)
    if worker:
        st.write(worker.info, worker.is_alive())

    with st.sidebar:
        is_running = worker is not None and worker.is_alive()
        is_ready = not is_running
        if st.button('Start worker', disabled=is_running):
            worker = Worker(name = worker_unique_name, routine = test_routine, daemon=True)
            ProcessWorkerStoreSingleton.set(worker)
            worker.start()
            st.experimental_rerun()
            
        if st.button('Stop worker', disabled=is_ready):
            worker.should_stop.set()
            worker.join()
            # worker = None
            ProcessWorkerStoreSingleton.set(worker)
            st.experimental_rerun()

    # if worker is None:
    #     st.markdown('No worker running.')
    # else:
    #     st.markdown(f'worker: {worker.pid}')
    #     placeholder = st.empty()
    #     while worker.is_alive():
    #         placeholder_container = placeholder.container()
    #         placeholder_container.markdown(f'status: {worker.info}')
    #         time.sleep(1)
    # st.info("worker is dead")
    # if worker:
    #     st.markdown(f'status: {worker.info}')
    #     time.sleep(2)
    #     st.markdown(f'status: {worker.info}')


    # if worker:
    #     st.markdown(f'status: {worker.status.value}')
    #     st.markdown(f'updated_at: {worker.updated_at.value}')
    #     st.markdown(f'message: {worker.message.value}')
    #     st.markdown(f'tracebacks: {worker.tracebacks[:]}')

if __name__ == '__main__':
    main()
