# Malaria Detection Web App

This project is a web application to detect malaria infection from blood cell images. It consists of a Next.js frontend and a Flask backend in the same repository.

## Project Structure

- `src/app/page.tsx`: Next.js frontend page with image upload and prediction display.
- `backend/app.py`: Flask backend API to handle image upload, feature extraction, and ML model prediction.
- `backend/feature_extraction.py`: Python module with feature extraction code.
- `model/model.pkl`: Placeholder for the trained ML model file (to be provided by the user).

## Setup and Run

### Backend

1. Create a Python virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install flask opencv-python-headless scikit-image pandas numpy joblib
```

3. Place your trained ML model file as `model/model.pkl`.

4. Run the Flask backend:

```bash
python backend/app.py
```

The backend will run on `http://localhost:5000`.

### Frontend

1. Install Node.js dependencies:

```bash
npm install
```

2. Run the Next.js frontend:

```bash
npm run dev
```

The frontend will run on `http://localhost:3000`.

## Usage

- Open the frontend in your browser.
- Upload a blood cell image.
- The image will be sent to the backend for processing and prediction.
- The prediction result will be displayed on the page.

## Notes

- The backend expects the ML model to be a scikit-learn compatible model saved with joblib.
- The feature extraction code uses watershed segmentation and texture features as provided.
- You may need to adjust the JSON data format in `backend/app.py` for your specific dataset.

## License

MIT License
