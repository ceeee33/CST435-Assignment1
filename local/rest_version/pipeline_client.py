import os
import time
import base64
import requests
from datetime import datetime
from PIL import Image
from threading import Thread

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:50051")

# Path to input folder
test_img_path = "test_image"

# List of images to process
images_to_process = [
    os.path.join(test_img_path, f)
    for f in os.listdir(test_img_path)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# Path to output folders
output_folder = "results"
timeline_folder = "timeline_plots"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(timeline_folder, exist_ok=True)

def format_timestamp(ts: float | None) -> str:
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def load_image(path: str) -> bytes:
    """Load image bytes or create a placeholder if not exists."""
    if not os.path.exists(path):
        img = Image.new('RGB', (100, 100), color='red')
        img.save(path)
    with open(path, 'rb') as f:
        return f.read()

def poll_status(task_id: str, file_name: str):
    """Poll the /status endpoint until the task completes."""
    while True:
        try:
            resp = requests.get(f"{ORCHESTRATOR_URL}/status/{task_id}", timeout=10)
            if resp.ok:
                data = resp.json()
                if data.get("success") or data.get("error"):
                    # Task finished
                    out_path = os.path.join(output_folder, f"processed_{file_name}")
                    with open(out_path, 'wb') as f:
                        f.write(base64.b64decode(data["processed_image_data"]))
                    print(f"[{file_name}] Saved processed image: {out_path}")

                    if data.get("timeline_plot_image"):
                        server_filename = data.get("timeline_plot_filename") or "pipeline_timeline.png"
                        base_name = os.path.basename(server_filename)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        client_plot_path = os.path.join(timeline_folder, f"{timestamp}_{base_name}")
                        plot_bytes = base64.b64decode(data["timeline_plot_image"])
                        with open(client_plot_path, "wb") as f:
                            f.write(plot_bytes)
                        print(f"[{file_name}] Timeline plot saved: {client_plot_path}")
                    else:
                        print("No timeline plot image ")
                        
                    if data.get("response"):
                        label = data["response"].get("label", "")
                        print(f"[{file_name}] Classification: {label}")

                    if data.get("error"):
                        print(f"[{file_name}] Server error: {data['error']}")

                    break  # Stop polling
                else:
                    # Pending
                    time.sleep(0.5)
            else:
                print(f"[{file_name}] Status HTTP error: {resp.status_code}")
                time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"[{file_name}] Polling request failed: {e}")
            time.sleep(1)

def process_image(image_path: str):
    """Send image to orchestrator and poll for result."""
    image_data = load_image(image_path)
    file_name = os.path.basename(image_path)
    start_time = time.time()
    print(f"Sending image: {file_name}")

    try:
        resp = requests.post(
            f"{ORCHESTRATOR_URL}/process",
            json={
                "image_data": base64.b64encode(image_data).decode("utf-8"),
                "file_name": file_name
            },
            timeout=10
        )

        if resp.ok:
            data = resp.json()
            if data.get("success"):
                task_id = data.get("task_id")
                print(f"[{file_name}] Task accepted with ID: {task_id}")
                # Start polling in this thread
                poll_status(task_id, file_name)
            else:
                print(f"[{file_name}] Server rejected task: {data.get('message')}")
        else:
            print(f"[{file_name}] HTTP error: {resp.status_code} - {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"[{file_name}] Request failed: {e}")
    except Exception as e:
        print(f"[{file_name}] Error: {e}")

def main():
    """Process multiple images concurrently with pipeline parallelism."""
    threads = []
    for img_path in images_to_process:
        t = Thread(target=process_image, args=(img_path,))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    print("\nAll images processed!")

if __name__ == '__main__':
    main()
