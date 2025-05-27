# Vidyasa Backend
Backend for IF4042 Information Retrieval System Final Project

## Overview
This repository contains the backend implementation for the Vidyasa Information Retrieval System, developed as the final project for IF4042 Information Retrieval System course.

## Tech Stack
- FastAPI
- Python 3.x
- (Add other relevant technologies)

## Getting Started

### Prerequisites
- Python 3.x
- pip

### Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/vidyasa-backend.git
cd vidyasa-backend
```

2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Get the dataset by running the following command:

Windows:
```powershell
New-Item -ItemType Directory -Path "data/raw" -Force | Out-Null
Invoke-WebRequest -Uri "https://drive.google.com/uc?export=download&id=1dDPHHzEuC3Zqy4csG2vk7XZ51JNTIRlD" -OutFile "data/IRCollection.zip"
Expand-Archive -Path "data/IRCollection.zip" -DestinationPath "data/raw/" -Force
Remove-Item "data/IRCollection.zip"
```
Linux/Mac:
```bash
mkdir -p data/raw && \
wget "https://drive.google.com/uc?export=download&id=1dDPHHzEuC3Zqy4csG2vk7XZ51JNTIRlD" -O "data/IRCollection.zip" && \
unzip -o data/IRCollection.zip -d data/raw/ && \
rm data/IRCollection.zip && \
```

5. run scripts

preprocess the dataset
```powershell
python scripts/preprocess.py
python scripts/project_setup.py
```

### Running the Application
```bash
uvicorn app.main:app --reload
```
The API will be available at http://localhost:8000

## API Documentation
Once the server is running, access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure
```
vidyasa-backend/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   ├── routers/
│   ├── services/
│   └── utils/
│
├── tests/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Reference
[Fast API tutorial](https://code.visualstudio.com/docs/python/tutorial-fastapi)