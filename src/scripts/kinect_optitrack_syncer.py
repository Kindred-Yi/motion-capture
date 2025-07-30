from datetime import datetime, timedelta
import pandas as pd
import argparse
import cv2
import math
import json
import chardet
import os
import shutil

# TODO Must change all the below to appropriate values
KINECT_START = r"D:/HAND_Human_Human_Study/Kinect/record_start_time_3pm_2_7_14.txt"
KINECT_OFFSET = 200600  # must change to correct OFFSET, use k4aviewer to find this
KINECT_VIDEO = r"D:/HAND_Human_Human_Study/Kinect/extracted videos/output_3pm_2_7_14.mkv"
OPTI_CSV = r"D:/HAND_Human_Human_Study/OptiTrack/peanut butter 3pm 7-14 2.csv"
OUTPUT_FOLDER = r"D:/HAND_Human_Human_Study/Kinect/mkv_2_colordepth/7-14 3pm 2"

# The global variables below probably will not change
KINECT_CUT_END_FRAMES = 20  # number of frames to cut at the end of the video, this is to remove the last few frames that are not needed
CONSTANT_OFFSET = .6        # offset in seconds, this is not constant for all videos, but constant for only this video
OPTI_FPS = 120                  # default number of frames on Optitrack
KINECT_FPS = 30                 # Current Kinect Script runs at 30 FPS

def read_timestamp_from_file(filepath: str, fmt: str = "%Y-%m-%d %H:%M:%S.%f") -> datetime:
    """
    Reads the first line of a text file and parses it into a datetime.
    Automatically detects encoding using chardet.

    Args:
        filepath (str): Path to the file containing the timestamp string.
        fmt (str): Format of the datetime string (default: full timestamp with microseconds)

    Returns:
        datetime: Parsed datetime object
    """
    # Detect encoding
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()
        result = chardet.detect(raw_bytes)
        encoding = result['encoding']
        if encoding is None:
            raise ValueError("Could not detect file encoding")

    # Decode and parse
    decoded_text = raw_bytes.decode(encoding).strip()
    return datetime.strptime(decoded_text, fmt)

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    # CAP_PROP_FRAME_COUNT returns a float, so cast to int
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def compute_opti_start_frame_and_time(target_time: datetime, base_start: datetime, fps: float):
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

def get_kinect_seconds(kinect_video_time_str):
    try:
        # 1. Split the string by the colon
        parts = kinect_video_time_str.split(':')

        # 2. Basic validation: check if there are exactly two parts
        if len(parts) == 2:
            minutes_str, seconds_str = parts

            # 3. Convert parts to integers
            minutes = int(minutes_str)
            seconds = int(seconds_str)

            # 4. Add more robust validation (optional but recommended)
            if not (0 <= minutes <= 599 and 0 <= seconds <= 59): # Example range for minutes, max 9:59:59 in 3 digits
                raise ValueError("Minutes should be 0-599 and seconds 0-59.")

            # 5. Calculate total seconds
            total_seconds = (minutes * 60) + seconds
            return total_seconds
        else:
            print(f"Error: Invalid time format '{kinect_video_time_str}'. Expected MM:SS.")

    except ValueError as e:
        print(f"Error converting time string '{kinect_video_time_str}': {e}. Please use MM:SS format.")

def compute_frame_num(total_seconds, kinect_start_frame, opti_start_frame):
    """
    given total_seconds into the kinect video as well as the calculated kinect_start_frame and opti_start_frame, 
    this method will return the kinect and optitrack frames that correspond to it
    Params:
        total_seconds : number of kinect seconds into the kinect video
        kinect_start_frame 
        opti_start_frame
    Returns:

    """
    kinect_frame = round(total_seconds * KINECT_FPS)
    if kinect_frame < kinect_start_frame:
        print("The given Kinect Video time is before there are optitrack frames. Please provide a minute and second pair that is greater than", kinect_start_time * KINECT_FPS, "seconds")
        exit(1)

    kinect_offset_frames = kinect_frame - kinect_start_frame
    opti_frame = opti_start_frame + kinect_offset_frames / KINECT_FPS * OPTI_FPS

    return kinect_frame, int(opti_frame)



if __name__ == "__main__":
    # 0. Parse command line arugment to convert time to frames
    parser = argparse.ArgumentParser(description='Gives option to convert kinect video time to kinect and optitrack frames')
    parser.add_argument(
        "-k",
        "--kinect_video_time",
        dest = 'kinect_video_time',
        type=str,
        default=None,
        help = 'Kinect video time in format MM:SS, e.g. 00:30 for 30 seconds'
    )
    parser.add_argument(
    "-c",
    "--create_csv",
    action="store_true",
    help="Flag to create CSV for Data Post Processing (PCA). Set to true if -c is included."
    )

    args = parser.parse_args()


    # 1. Load and offset Kinect start time as well as count frames in the video
    try:
        kinect_start_time = read_timestamp_from_file(KINECT_START)
    except Exception as e:
        print(f"Failed to read Kinect start time: {e}")
        exit(1)
    
    kinect_start_time += timedelta(microseconds=KINECT_OFFSET) + timedelta(seconds=CONSTANT_OFFSET)
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
        opti_start_frame, new_opti_start_time, final_offset = compute_opti_start_frame_and_time(
            kinect_start_time, opti_start_time, OPTI_FPS
        )
    else:
        print("Kinect started first logic")
        new_kinect_start_frame = math.ceil(initial_start_offset.total_seconds() * KINECT_FPS) # round up so kinect will always be within the optitrack time
        new_kinect_start_time = new_kinect_start_time + timedelta(seconds=kinect_start_frame / KINECT_FPS) # adds the time 
        opti_start_frame, new_opti_start_time, final_offset = compute_opti_start_frame_and_time(
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
    JSON_FILE = os.path.join(OUTPUT_FOLDER, "frame_indices.json")

    with open(JSON_FILE, 'w') as f:
        json.dump(frame_data, f, indent=4)
    print(f"Frame indices saved to {JSON_FILE}")

    # 10. Calculate frame for given kinect video time

    kinect_video_time_str = args.kinect_video_time
    total_seconds = None # Initialize to None in case of no input or error

    if kinect_video_time_str is not None:
        total_seconds = get_kinect_seconds(kinect_video_time_str)
        kinect_frame, opti_frame = compute_frame_num(total_seconds=total_seconds, kinect_start_frame=kinect_start_frame, opti_start_frame=opti_start_frame)
        print(f"At Kinect Video Time {kinect_video_time_str}\nKinect Frame: {kinect_frame}, Optitrack Frame: {opti_frame}")
    else:
        print("No kinect_video_time provided.")
    
    # 11. Create CSV for Data Post Processing (PCA)
    if args.create_csv:
        # Step 1: Copy the original file (used to avoid modifying the original while testing)
        copied_file = os.path.join(OUTPUT_FOLDER, "calib_desktop_000_copy.csv")
        shutil.copyfile(OPTI_CSV, copied_file)

        # Step 2: Read metadata rows (first 6 lines)
        with open(copied_file, "r", encoding="utf-8", errors="ignore") as f:
            metadata_lines = [next(f) for _ in range(6)]
        
        # Add two empty columns (,,) to the start of each metadata line to account for tghe two new columsn added to data frame for kinect frame and time
        metadata_lines = [',,' + line for line in metadata_lines]

        # Step 3: Read the rest as a DataFrame starting from header row (line 7)
        df = pd.read_csv(copied_file, skiprows=6, header=0, low_memory=False)

        # Step 4: Rename columns
        df.columns = df.columns.str.strip()
        df.rename(columns={
            "Frame": "Optitrack Frame",
            "Time (Seconds)": "Optitrack Seconds"
        }, inplace=True)

        # ✅ Ensure "Optitrack Frame" is numeric so we can filter by frame number
        df["Optitrack Frame"] = pd.to_numeric(df["Optitrack Frame"], errors="coerce")

        # ✅ Drop any rows where "Optitrack Frame" was invalid (NaN after conversion)
        df.dropna(subset=["Optitrack Frame"], inplace=True)

        # ✅ Convert to integers for precise matching (optional, for safety)
        df["Optitrack Frame"] = df["Optitrack Frame"].astype(int)

        # ✅ Keep only rows within the synchronized time window
        df = df[
            (df["Optitrack Frame"] >= opti_start_frame) &
            (df["Optitrack Frame"] <= opti_end_frame)
        ].reset_index(drop=True)

        # Step 5: Filter rows (keep every 4th)
        df.reset_index(drop=True, inplace=True)
        df = df[df.index % 4 == 0].reset_index(drop=True)

        # Step 6: Add Kinect columns
        # ✅ Update Kinect Frame and Time using actual aligned start frame
        num_rows = len(df)
        kinect_frames = list(range(kinect_start_frame, kinect_start_frame + num_rows))
        df["Kinect Frame"] = kinect_frames
        df["kinect time (s)"] = [(f / KINECT_FPS) for f in kinect_frames]


        # Step 7: Reorder columns
        df = df[["Kinect Frame", "kinect time (s)"] + [col for col in df.columns if col not in ["Kinect Frame", "kinect time (s)"]]]

        # Ensure unique column names by removing duplicates
        df.columns = pd.Series(df.columns).apply(lambda x: x.split('.')[0]).tolist()

        # Rename columns starting with "Unnamed" to an empty string
        df.columns = ['' if str(col).startswith('Unnamed') else col for col in df.columns]

        # Step 8: Save metadata + modified data to a new file
        output_file = os.path.join(OUTPUT_FOLDER, "processed.csv")
        with open(output_file, "w", encoding="utf-8", newline='') as f:
            f.writelines(metadata_lines)        # Metadata (6 lines)
            df.to_csv(f, index=False, lineterminator='\n')  # ✅ correct
        print(f"Processed CSV saved to {output_file}")


    






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
