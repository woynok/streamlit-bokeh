import time
import pandas as pd
import random
import streamlit as st
from lib.backgroundjob.process_workers import Worker, ProcessWorkerStoreSingleton, WorkerInfo

from lib.backgroundjob.test_routine import test_routine
# worker_name = "my-name"
worker_name = st.sidebar.selectbox("worker name", ["my-name", "my-name2"])
worker: Worker = ProcessWorkerStoreSingleton.get(worker_name)
st.write(ProcessWorkerStoreSingleton.d_workers)
# if worker:
#     st.write(worker.info, worker.is_alive())

with st.sidebar:
    is_running = worker is not None and worker.is_alive()
    is_ready = not is_running
    if st.button('Start worker', disabled=is_running):
        worker = ProcessWorkerStoreSingleton.start_worker(worker_name = worker_name, routine = test_routine)
        st.experimental_rerun()
        
    if st.button('Stop worker', disabled=is_ready):
        worker.should_stop.set()
        worker.join()
        # worker = None
        ProcessWorkerStoreSingleton.set(worker)
        st.experimental_rerun()

infos = ProcessWorkerStoreSingleton.get_all_worker_info_list()
for info in infos:
    st.write(info)

def construct_streamlit_data_editor_column_configs():
    from streamlit.column_config import CheckboxColumn, TextColumn, ImageColumn
    
    column_configs = {
        "cancel" : CheckboxColumn(
            label = "中止する",
            # width = 10,
        ),
        "name" : TextColumn(
            label = "名前",
            # width = 30,
        ),
        "updated_at" : TextColumn(
            label = "最終更新日時",
            # width = 60,
        ),
        "last_status" : TextColumn(
            label = "最後の状態",
            # width = 60,
        ),
        "is_alive" : TextColumn(
            label = "実行中か",
            # width = 60,
        ),
    }
    return column_configs
column_configs = construct_streamlit_data_editor_column_configs()
def running_bool_to_emoji(x):
    running_icons_vehicle = ["🚗", "🚕", "🚙", "🚌"]
    running_icon = running_icons_vehicle[0]
    if x:
        return running_icon
    else:
        return "－"

if infos:
    df_infos = pd.DataFrame(infos)
    df_infos["cancel"] = [False] * len(infos)
    df_infos["is_alive"] = df_infos["is_alive"].apply(running_bool_to_emoji)
    df_infos = df_infos[column_configs.keys()]
    df_infos_result = st.data_editor(df_infos, column_config=column_configs)
    
    cancel_worker_names = df_infos_result[df_infos_result["cancel"]]["name"].tolist()
    if cancel_worker_names:
        placeholder_button = st.empty()
        if placeholder_button.button("中止する"):
            placeholder_button.info("中止中...")
            for worker_name in cancel_worker_names:
                ProcessWorkerStoreSingleton.stop(worker_name)
            st.experimental_rerun()
# df_infos_cancel = st.data_editor(
#     df_infos,
#     column_config=column_configs
# )


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

# if __name__ == '__main__':
#     main()
