import os
import time
import grpc
from datetime import datetime
from PIL import Image
from threading import Thread

import pipeline_pb2
import pipeline_pb2_grpc

ORCHESTRATOR_ADDR = os.getenv("ORCHESTRATOR_ADDR", "localhost:50030")

# Path to input folder
test_img_path = "test_image"

# List of images to process
images_to_process = [
    os.path.join(test_img_path, f)
    for f in os.listdir(test_img_path)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# Path to output folder
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

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

def process_image(stub, image_path: str):
    image_data = load_image(image_path)
    file_name = os.path.basename(image_path)
    start_time = time.time()
    start_ts = format_timestamp(start_time)
    print(f"Sending image: {file_name} at {start_ts}")
    
    resp = stub.Process(pipeline_pb2.ProcessRequest(image_data=image_data, file_name=file_name))
    elapsed = (time.time() - start_time) * 1000

    if resp.success:
        out_path = os.path.join(
            output_folder,
            f"processed_{os.path.splitext(file_name)[0]}.png"
        )
        # out_path = f"processed_{os.path.splitext(file_name)[0]}.png"
        with open(out_path, 'wb') as f:
            f.write(resp.processed_image_data)
        print(f"[{file_name}] Classification: {resp.classification_label}")
        print(f"[{file_name}] Server-side processing time: {resp.response_time_ms:.2f} ms")
        print(f"[{file_name}] Client total time: {elapsed:.2f} ms")
        print(f"[{file_name}] Saved: {out_path}")

        if resp.timeline_plot_image:
            os.makedirs("timeline_plots", exist_ok=True)
            server_filename = resp.timeline_plot_filename or "pipeline_timeline.png"
            base_name = os.path.basename(server_filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            client_plot_path = os.path.join("timeline_plots", f"{timestamp}_{base_name}")
            with open(client_plot_path, "wb") as f:
                f.write(resp.timeline_plot_image)
            print(f"[{file_name}] Timeline plot saved: {client_plot_path}")

    else:
        print(f"[{file_name}] Processing failed: {resp.message}")

def main():
    channel = grpc.insecure_channel("localhost:50030")
    stub = pipeline_pb2_grpc.ImageProcessorOrchestratorStub(channel)

    threads = []
    for img_path in images_to_process:
        t = Thread(target=process_image, args=(stub, img_path))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
        

if __name__ == '__main__':
    main()


