import subprocess
from datetime import datetime
from heart_beat import HeartBeat
from pytz import timezone
import time
import streamlit as st
timezone_tokyo = timezone("Asia/Tokyo")

if st.button("subprocess start"):
    subprocess.run(["python", "routine_example.py"])

def very_slow_routine(coefficient = 1.0e-4):
    print(f"{datetime.now(tz=timezone_tokyo)}::very_slow_routine() starting...")
    for i in range(20000):
        for j in range(10000):
            c = i * j
    print(f"{datetime.now(tz=timezone_tokyo)}::very_slow_routine() done...")
    return c * coefficient

placeholder_button = st.empty()
button = placeholder_button.button("start", key="start_initial")
def save_result(result):
    print("saving result....")
    time.sleep(3)
    with open("./session_state.txt", "w") as f:
        f.write(f"{result}")
        f.write("\n")
        f.write(f"this is saved at {datetime.now(tz=timezone_tokyo).strftime('%Y-%m-%d %H:%M:%S')}")
    print("result saved.")

heart_beat = HeartBeat(
    pulse_duration = 1.5,
    routine = very_slow_routine,
    message = "very_slow_routine",
    save_result=save_result,
    )

if button:
    placeholder_button.button("start", disabled=True, key="start_disabled")
    with st.spinner():
        with heart_beat.drum(coefficient = 1.0e-8) as vibes:
            result = vibes.result
    st.write(result)
    placeholder_button.button("やりなおす")
