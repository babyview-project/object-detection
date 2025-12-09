import pandas as pd

def get_metadata(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get number of unique videos
    unique_videos = df['superseded_gcp_name_feb25'].nunique()
    
    # Get number of unique frames
    unique_frames = df['original_frame_path'].nunique()
    
    # # Get number of unique subjects by splitting video_id
    # # Assuming format is "subject_restofid"
    # unique_subjects = df['superseded_gcp_name_feb25'].str.split('_').str[0].nunique()
    # print(unique_subjects)
    
    # Print results
    print(f"Number of unique videos: {unique_videos}")
    print(f"Number of unique frames: {unique_frames}")
    # print(f"Number of unique subjects: {unique_subjects}")
    
    return unique_videos, unique_frames #, unique_subjects

if __name__ == "__main__":
    # Replace with your CSV file path
    csv_path = "../allframes_1fps.csv"
    get_metadata(csv_path)
