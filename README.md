# Signature Verification

A full-stack signature verification system with an ML backend and a React frontend.

## Project Structure

```
signature_verification/
├── ml/               # Python/Flask ML backend
│   ├── models/       # Trained model files (.h5, .pth)
│   ├── data/         # Training & test datasets
│   ├── src/
│   │   ├── app.py    # Flask API entry point (port 5000)
│   │   └── ...       # Training, preprocessing, inference scripts
│   └── requirements.txt
├── website/          # React (Vite) frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── Center.jsx   # Main upload & analyze UI
│   │   └── pages/
│   └── package.json
└── README.md
```

## Getting Started

### 1. Start the ML Backend

```bash
cd ml

# Create a virtual environment (once)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask API
cd src
python app.py
```

The API will be available at **http://localhost:5000**.

### 2. Start the React Frontend

```bash
cd website

# Install dependencies (once)
npm install

# Start the dev server
npm run dev
```

The frontend will be available at **http://localhost:5173** (default Vite port).

### API Endpoint

**POST /predict** — Upload two signature images for comparison.

| Field    | Type | Description          |
| -------- | ---- | -------------------- |
| `image1` | file | First signature image  |
| `image2` | file | Second signature image |

**Response:**
```json
{
  "result": "Genuine",
  "confidence": 0.92,
  "match_percentage": 92.0
}
```
