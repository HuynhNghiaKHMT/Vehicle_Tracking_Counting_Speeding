# your_project/config/initial_values.py

def get_default_session_state_values():
    """Returns a dictionary of default Streamlit session state values."""
    return {
        'run_processing': False,
        'preview_frame': None,
        'video_dims': (0, 0), # (width, height)
        'temp_video_path': None,
        'last_uploaded_file_id': None,
        'line_coords_config': None,
        'source_points_config': None,
        'target_width_real_config': 32,
        'target_height_real_config': 50,
    }