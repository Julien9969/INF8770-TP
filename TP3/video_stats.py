import cv2
import os

def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get frame properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate duration in seconds
    duration_sec = frame_count / fps

    # Release the video capture object
    cap.release()

    return fps, width, height, duration_sec

def analyze_videos_in_folder(folder_path):
    # Get list of all files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("No video files found in the specified folder.")
        return

    # Initialize lists to store video properties
    fps_list = []
    width_list = []
    height_list = []
    duration_list = []

    # Iterate through each video file
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"Analyzing {video_file}...")

        # Get video properties
        fps, width, height, duration_sec = get_video_info(video_path)

        # Append properties to lists
        fps_list.append(fps)
        width_list.append(width)
        height_list.append(height)
        duration_list.append(duration_sec)

    # Calculate average FPS
    if fps_list:
        average_fps = sum(fps_list) / len(fps_list)
    else:
        average_fps = 0

    # Display results
    print("\nAnalysis Summary:")
    print(f"Total Videos Analyzed: {len(video_files)}")
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Average Frame Width: {sum(width_list) / len(width_list):.2f}")
    print(f"Average Frame Height: {sum(height_list) / len(height_list):.2f}")
    print(f"Total Duration (in seconds): {sum(duration_list):.2f}")

if __name__ == "__main__":
    analyze_videos_in_folder("C:/Users/Alexandre/Desktop/INF8770-TP/TP3/data/mp4")
