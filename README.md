# Diabetic-Retinopathy-detect-project_Web-service

Web service implementation for DR prediction.
This repository contains frontend/backend integration files centered on a Streamlit app.

## Repository Scope
- `app.py`: Streamlit inference app
- `best_model_recall.pth`: trained DR model weight file
- `requirements.txt`: Python dependencies
- `DR_model_heatmap_Grad-CAM_map.ipynb`, `DR_Model_Visualization.ipynb`: model visualization notebooks
- `Source_image(input).png`, `Source_image(Output).png`: example assets

## Requirements
- Python 3.10+
- Git LFS (for large model file handling)

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

After startup, open the local URL shown by Streamlit (default: `http://localhost:8501`).

## Notes
- The app expects `best_model_recall.pth` in the repository root.
- If cloning fresh, run `git lfs pull` to fetch large model files.
