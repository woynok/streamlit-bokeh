# from dataclasses import dataclass
# import multiprocessing
# import time
# import streamlit as st

# class Worker(multiprocessing.Process):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.should_stop = multiprocessing.Event()
#         self.counter = multiprocessing.Value('i', 0)
        
#     def run(self):
#         def heavy_computation():
#             sum = 0
#             for i in range(10000):
#                 sum += 1.0/((i + 1) * (i + 2))
#             return sum

#         while not self.should_stop.wait(0):
#             heavy_computation()
#             self.counter.value += 1

# @dataclass
# class ProcessManager:
#     worker = None
    
#     def get_worker(self):
#         return self.worker
    
#     def is_running(self):
#         return self.worker is not None and self.worker.is_alive()
    
#     def start_worker(self):
#         if self.worker is not None and self.worker.is_alive():
#             self.stop_worker()
#         self.worker = Worker(daemon=True)
#         self.worker.start()
#         return self.worker
    
#     def stop_worker(self):
#         self.worker.should_stop.set()
#         self.worker.join()
#         self.worker = None

# def main():
#     process_manager = ProcessManager()
#     st.write(id(process_manager))
#     with st.sidebar:
#         if st.button('Start worker', disabled=process_manager.is_running()):
#             worker = process_manager.start_worker()
#             st.experimental_rerun()
            
#         if st.button('Stop worker', disabled=not process_manager.is_running()):
#             process_manager.stop_worker()
#             st.experimental_rerun()
    
#     if not process_manager.is_running():
#         st.markdown('No worker running.')
#     else:
#         worker = process_manager.get_worker()
#         st.markdown(f'worker: {worker.name}')
#         placeholder = st.empty()
#         while worker.is_alive():
#             placeholder.markdown(f'counter: {worker.counter.value}')
#             time.sleep(1)

#     # 別セッションでの更新に追従するために、定期的にrerunする
#     # time.sleep(1)
#     st.experimental_rerun()

# if __name__ == '__main__':
#     main()
