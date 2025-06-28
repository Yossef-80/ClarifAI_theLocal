import multiprocessing as mp
from AllModels_Tracking_Enhanced_5_frame import detect_faces_with_tracking

video_path = "Source Video/S 8 Marta - Kindergarten Theater with Tanya-2m.mkv"

def run_instance(instance_id):
    window_name = f"Tracked Faces {instance_id}"
    output_path = f"output_{instance_id}.mp4"
    print(f"🔄 Starting instance {instance_id}")
    detect_faces_with_tracking(video_path,  window_name)
    print(f"✅ Finished instance {instance_id}")

if __name__ == '__main__':
    processes = []
    for i in range(3):
        p = mp.Process(target=run_instance, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
