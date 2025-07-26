An AI-powered tool for breast cancer detection using CNNs to classify ultrasound images. Features Grad-CAM heatmaps, a Flask API, and a React frontend, deployed via Docker. #MedicalImaging #MachineLearning

Installation

Clone: `git clone https://github.com/kriyanshi21/BreastCancerVision.git\`
Run: `docker-compose up --build`
Usage

Web: Visit `http://localhost:3000\` to upload images.
API: `curl -X POST -F "image=@test.jpg" http://localhost:5000/predict\`
Model Download

Download `model.h5` from [Google Drive link] and place in `backend/`.