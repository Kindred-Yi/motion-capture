from datetime import datetime, timedelta
import pandas as pd
import argparse
import cv2
import math
import json

# TODO Must change all the below to appropriate values
KINECT_START = r"D:/HAND_Human_Human_Study/Kinect/start_time_16am_3_710.txt"
KINECT_OFFSET = 200388                                                         # must change to correct OFFSET, use k4aviewer to find this
KINECT_VIDEO = r"D:/HAND_Human_Human_Study/Kinect/output_16am_3_710.mkv"
KINECT_CUT_END_FRAMES = 20  # number of frames to cut at the end of the video, this is to remove the last few frames that are not needed
OPTI_CSV = r"D:/HAND_Human_Human_Study/OptiTrack/peanut butter 4pm 7-10 3.csv"
OUTPUT_FILE = r"D:/HAND_Human_Human_Study/Kinect/mkv_2_colordepth/7-10 4pm 3"

# The global variables below probably will not change
OPTI_FPS = 120                  # default number of frames on Optitrack
KINECT_FPS = 30                 # Current Kinect Script runs at 30 FPS

def read_timestamp_from_file(filepath: str, fmt: str = "%Y-%m-%d %H:%M:%S.%f") -> datetime:
    """
    Reads the first line of a text file and parses it into a datetime.
    The file should contain a single timestamp string matching `fmt`.
    """
    with open(filepath, 'r', encoding='utf-16') as f:
        line = f.readline().strip()
    return datetime.strptime(line, fmt)

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    # CAP_PROP_FRAME_COUNT returns a float, so cast to int
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def compute_start_frame_and_time(target_time: datetime, base_start: datetime, fps: float):
    """
    Choose floor or ceil of the frame index that best matches target_time.
    Returns: (frame_idx, matched_time, offset)
    """
    offset = target_time - base_start
    total = offset.total_seconds() * fps
    floor_frame = math.floor(total)
    ceil_frame = math.ceil(total)
    time_floor = base_start + timedelta(seconds=floor_frame / fps)
    time_ceil = base_start + timedelta(seconds=ceil_frame / fps)
    # pick closer
    if abs((target_time - time_floor).total_seconds()) <= abs((target_time - time_ceil).total_seconds()):
        return floor_frame, time_floor, offset
    return ceil_frame, time_ceil, offset

if __name__ == "__main__":
    # 1. Load and offset Kinect start time as well as count frames in the video

    try:
        kinect_start_time = read_timestamp_from_file(KINECT_START)
    except Exception as e:
        print(f"Failed to read Kinect start time: {e}")
        exit(1)
    
    kinect_start_time += timedelta(microseconds=KINECT_OFFSET)
    print("Kinect start time from file:", kinect_start_time)

    kinect_total_frames = count_frames(KINECT_VIDEO)
    print(f"Found {kinect_total_frames} Kinect frames in {KINECT_VIDEO}")

    # 1) Read the first line of the file as raw text
    with open(OPTI_CSV, 'r') as f:
        first_line = f.readline().strip().split(',')

    # 2) Locate “Capture Start Time” and “Total Frames in Take”
    i_start   = first_line.index('Capture Start Time')    # → 10
    i_frames  = first_line.index('Total Frames in Take')  # → 14

    opti_start_time_str = first_line[i_start + 1]   # e.g. "2025-07-10 04.03.57.425 PM"
    opti_total_frames   = int(first_line[i_frames + 1])  # e.g. 16844

    # 3) (Optional) Turn that start‑time string into a pandas Timestamp
    #    Note: the “PM” means you need %I for 12‑hour clock and %p for AM/PM
    opti_start_time = pd.to_datetime(
        opti_start_time_str,
        format="%Y-%m-%d %I.%M.%S.%f %p"
    )

    print("Optitrack Capture start time:", opti_start_time)
    print("Optitrack Total frames in take:", opti_total_frames)

    # 4) Calculate the end times
    opti_video_len = opti_total_frames / OPTI_FPS
    opti_end_time = opti_start_time + timedelta(seconds=opti_video_len)


    kinect_video_len = kinect_total_frames / KINECT_FPS
    kinect_end_time = kinect_start_time + timedelta(seconds=kinect_video_len)

    # 3. Compute clock offset
    initial_start_offset = kinect_start_time - opti_start_time
    print(initial_start_offset.total_seconds())

    # 4) checks which video started first and then adjusts start times accordingly
    kinect_start_frame = 0    
    opti_start_frame = 0

    new_kinect_start_time = kinect_start_time
    new_opti_start_time = opti_start_time

    if initial_start_offset.total_seconds() >= 0:
        print("OptiTrack started first logic")
        kinect_start_frame = 0
        opti_start_frame, new_opti_start_time, final_offset = compute_start_frame_and_time(
            kinect_start_time, opti_start_time, OPTI_FPS
        )
    else:
        print("Kinect started first logic")
        new_kinect_start_frame = math.ceil(initial_start_offset.total_seconds() * KINECT_FPS)
        new_kinect_start_time = new_kinect_start_time + timedelta(seconds=kinect_start_frame / KINECT_FPS)
        opti_start_frame, new_opti_start_time, final_offset = compute_start_frame_and_time(
            new_kinect_start_time, opti_start_time, OPTI_FPS
        )
        
    
    print("kinect start frame:", kinect_start_frame)
    print("optitrack start frame:", opti_start_frame, "\n")
    
    # 5. checks with video ends first and then adjusts the end time accordingly
    initial_end_offset = kinect_end_time - opti_end_time

    kinect_end_frame = kinect_total_frames
    opti_end_frame = opti_total_frames

    if initial_end_offset.total_seconds() >= 0:
        print("Optitrack ends first")
        new_kinect_len= opti_end_time - kinect_start_time
        kinect_end_frame = math.floor(new_kinect_len.total_seconds() * KINECT_FPS) - KINECT_CUT_END_FRAMES

        new_kinect_end_time = kinect_start_time + timedelta(seconds= kinect_end_frame / KINECT_FPS)
        opti_end_delta_time = new_kinect_end_time - opti_start_time
        print(opti_end_delta_time)
        opti_end_frame = math.floor(opti_end_delta_time.total_seconds() * OPTI_FPS)

    else:
        print("Kinect ends first Logic")
        
        kinect_end_frame = kinect_end_frame - KINECT_CUT_END_FRAMES# does not change, only including this code to improve readability
        opti_end_delta_time = kinect_end_time - opti_start_time
        opti_end_frame = math.floor(opti_end_delta_time.total_seconds() * OPTI_FPS)

    print("kinect end frame:", kinect_end_frame)
    print("optitrack end frame:", opti_end_frame)

    # 9. Save start/end frames to JSON
    frame_data = {
        "kinect_start_frame": kinect_start_frame,
        "kinect_end_frame": kinect_end_frame,
        "optitrack_start_frame": opti_start_frame,
        "optitrack_end_frame": opti_end_frame
    }
    JSON_FILE = OUTPUT_FILE + "frame_indices.json"
    with open(JSON_FILE, 'w') as f:
        json.dump(frame_data, f, indent=4)
    print(f"Frame indices saved to {OUTPUT_FILE}")




    # # 6. Build Kinect-frame timestamps at nominal 30 FPS
    # fps_kinect = 30.0
    # kine_times = [
    #     kinect_start + timedelta(seconds=i/fps_kinect)
    #     for i in range(n_kinect)
    # ]
    # df_kine = pd.DataFrame({
    #     'kinect_frame': range(n_kinect),
    #     'raw_kine_time': kine_times
    # })

    # # 7. Align Kinect times into OptiTrack's timebase
    # df_kine['kine_time'] = df_kine['raw_kine_time'] - offset

    # # 8. Nearest-neighbor join
    # df_kine = df_kine.sort_values('kine_time')
    # df_opti = df_opti.sort_values('opti_time')
    # df_sync = pd.merge_asof(
    #     df_kine, df_opti,
    #     left_on='kine_time', right_on='opti_time',
    #     direction='nearest'
    # )

    # # 9. Extract and print your frame-to-frame mapping
    # mapping = df_sync[['kinect_frame','opti_frame']]
    # print(mapping.head(10))
