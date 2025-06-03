# your_project/utils/app_state_utils.py
import streamlit as st
import os
from config.initial_values import get_default_session_state_values

def reset_session_state_and_ui(video_placeholder, video_info_placeholder):
    """Resets all relevant session state variables and UI elements."""
    st.session_state.run_processing = False
    
    # Clean up temporary video file if it exists
    if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
        try:
            os.remove(st.session_state.temp_video_path)
        except PermissionError:
            st.warning("Could not remove temp video file. It might still be in use. Please restart the app if issues persist.")
        finally:
            st.session_state.temp_video_path = None
    
    # Reset other session state variables to their default values
    default_values = get_default_session_state_values()
    for key, value in default_values.items():
        st.session_state[key] = value
    
    # NEW: Also reset vehicle_history_data
    st.session_state.vehicle_history_data = {}

    # Clear UI placeholders
    video_placeholder.empty()
    video_info_placeholder.info("Upload a video to see its dimensions.")