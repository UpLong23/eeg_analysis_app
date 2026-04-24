import streamlit as st
import pandas as pd
from scipy import signal
from io import BytesIO

st.set_page_config(page_title='Load data', layout='wide')

CHANNELS = ['Fz','C3','Cz','C4','Pz','PO7','OZ','PO8']

if "uploaded_eeg_files" not in st.session_state:
    st.session_state.uploaded_eeg_files = {}

if "selected_eeg_name" not in st.session_state:
    st.session_state.selected_eeg_name = None

if "eeg" not in st.session_state:
    st.session_state.eeg = None

if "filtered_eeg" not in st.session_state:
    st.session_state.filtered_eeg = None
    
if "cut" not in st.session_state:
    st.session_state.cut = None

#-----------------------------------------------

if "uploaded_log_files" not in st.session_state:
    st.session_state.uploaded_log_files = {}
    
if "selected_log_name" not in st.session_state:
    st.session_state.selected_log_name = None
    
if "log" not in st.session_state:
    st.session_state.log = None
    
if "transformed_log" not in st.session_state:
    st.session_state.transformed_log = None



@st.cache_data
def load_data(file_bytes):
    eeg = pd.read_csv(BytesIO(file_bytes), header=None)
    for i, ch in enumerate(CHANNELS):
        eeg.rename(columns={i: ch}, inplace=True)
    eeg.rename(columns={16: 'Q'}, inplace=True)
    return eeg[CHANNELS]

def load_logs(file_bytes):
    log = pd.read_csv(BytesIO(file_bytes), parse_dates=['timestamp'])
    return log

def save_uploaded_files():
    files = st.session_state.eeg_uploader
    if files:
        for f in files:
            st.session_state.uploaded_eeg_files[f.name] = load_data(f.getvalue())
            
def save_uploaded_logs():
    files = st.session_state.log_uploader
    if files:
        for f in files:
            print(f)
            st.session_state.uploaded_log_files[f.name] = load_logs(f.getvalue())

def filter_eeg(eeg):
    eeg_bs = eeg.copy()
    eeg_bs = eeg_bs - eeg_bs.mean(axis=0)

    cutoffs = [1, 45]
    fs = 250
    order = 2

    b, a = signal.butter(
        order,
        [cutoffs[0]/(fs/2), cutoffs[1]/(fs/2)],
        btype="bandpass"
    )

    for column in eeg_bs.columns:
        eeg_bs[column] = signal.filtfilt(b, a, eeg_bs[column])

    st.session_state.filtered_eeg = eeg_bs
    st.write(st.session_state.filtered_eeg)

def transform_log(log):
    deltas            = log['timestamp'] - log['timestamp'][0]
    log['seconds']    = deltas.dt.total_seconds()
    log['stim_start'] = log['seconds'] + 0.5
    log['stim_stop']  = log['seconds'] + 0.5 + 0.6
    
    st.session_state.transformed_log = log

with st.expander("EEG Data", expanded=True):
    with st.expander("Upload", expanded=False):
        st.file_uploader(
            label="Load EEG data",
            accept_multiple_files=True,
            type="csv",
            key="eeg_uploader",
            on_change=save_uploaded_files
        )

    available_files = list(st.session_state.uploaded_eeg_files.keys())

    if available_files:
        if st.session_state.selected_eeg_name not in available_files:
            st.session_state.selected_eeg_name = available_files[0]

        selected_name = st.selectbox(
            "Select raw EEG file",
            available_files,
            index=available_files.index(st.session_state.selected_eeg_name)
        )

        st.session_state.selected_eeg_name = selected_name

        eeg = st.session_state.uploaded_eeg_files[selected_name]

        cutoff = st.number_input(
            "Insert cutoff in seconds",
            min_value=0.0,
            max_value=None,
            value=0.0,
            step=1.0,
            # width=100,
            format="%.2f"
        )

        eeg_cut = eeg[int(cutoff / 4e-3):]
        eeg_cut = eeg_cut.reset_index(drop=True)
        st.session_state.eeg = eeg_cut
        st.session_state.cut = cutoff

        st.write(eeg_cut)

        st.button(
            "Filter Data",
            on_click=filter_eeg,
            args=(st.session_state.eeg,)
        )
        

    if st.session_state.filtered_eeg is not None:
        st.write(st.session_state.filtered_eeg)
        
        
        


with st.expander("LOG Data", expanded=True):
    with st.expander("Upload", expanded=False):
        st.file_uploader(
            label="Load LOG data",
            accept_multiple_files=True,
            type="csv",
            key="log_uploader",
            on_change=save_uploaded_logs
        )

    available_logs = list(st.session_state.uploaded_log_files.keys())

    if available_logs:
        if st.session_state.selected_log_name not in available_logs:
            st.session_state.selected_log_name = available_logs[0]

        selected_name_log = st.selectbox(
            "Select LOG file",
            available_logs,
            index=available_logs.index(st.session_state.selected_log_name)
        )

        st.session_state.selected_log_name = selected_name_log

        log = st.session_state.uploaded_log_files[selected_name_log]

        st.session_state.log = log

        st.write(log)

        st.button(
            "Transform LOG",
            on_click=transform_log,
            args=(st.session_state.log,)
        )
        
    # if st.session_state.transformed_log is not None:
    #     st.write(st.session_state.transformed_log)
    
        