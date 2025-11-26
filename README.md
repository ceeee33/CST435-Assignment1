# 📘 Microservices Image Processing Pipeline — REST & gRPC

This repository contains a fully containerized **microservices-based image processing pipeline**, implemented in both **REST** and **gRPC** versions.

The system processes images through several independent microservices:

- Uploader Service  
- Noise Reduction Service  
- Background Removal Service  
- Classifier Service  

Each service runs independently in a Docker container, allowing scalable deployment on **single-machine** or **multi-machine (distributed)** setups.

---

## 🚀 Features

### **REST Version**
- Communication through HTTP REST APIs.
- Simple and beginner-friendly to test.

### **gRPC Version**
- Faster binary serialization using Protocol Buffers.
- Lower latency and better distributed performance.

### **Modular Microservices**
Each microservice performs a specific task:
- **Uploader** — Receives images and forwards them.
- **Noise Reducer** — Removes noise from images.
- **Background Removal** — Extracts foreground via segmentation.
- **Classifier** — Predicts the class label.

### **Deployment Options**
- **Single Machine:** All services run locally.
- **Distributed Mode:** Each service runs on different machines.

---

### **🔧 1. Running the REST Version**

```bash
cd multiple/rest_version
docker compose up -d --build
```

View live logs:
```bash
docker compose logs -f
```

### **🔧 1. Running the GRPC Version**
```bash
cd multiple/grpc_version
docker compose up -d --build
```

View live logs:
```bash
docker compose logs -f
```

### **🌐 Running an Individual Service**

Example: Run Background Removal (REST):

```bash
docker build -t rest-multiple-background .
docker run -d -p 50062:50062 --name rest-multiple-background rest-multiple-background
```

List Docker images:
```bash
docker images
```
## **📦 Technologies Used**

- Docker & Docker Compose
- Python (Flask / FastAPI for REST)
- gRPC & Protocol Buffers
- OpenCV / Image Processing Libraries
- Distributed System Techniques
- Logging & Benchmarking Tools
