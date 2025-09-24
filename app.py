import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import gradio as gr
import pickle
import os

# Path configurations
MODEL_PATH = "coimbra_dnn_model.h5"
SCALER_PATH = "scaler.pkl"

# Check if model exists, if not, we'll train it
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        # Load pre-trained model and scaler
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    else:
        # If files don't exist, train from scratch
        return train_model()

def train_model():
    print("Training new model...")
    # Load the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv"
    df = pd.read_csv(url)
    
    # Split dataset into features (X) and target (y)
    X = df.drop("Classification", axis=1)
    y = df["Classification"] - 1  # Adjust labels to be 0 and 1
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define DNN Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=X.shape[1], activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_scaled, y, epochs=50, batch_size=8, verbose=1)
    
    # Save the trained model and scaler
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    
    return model, scaler

# Load or train the model
model, scaler = load_or_train_model()

# Define prediction function
def predict(age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp_1):
    # Create input array from parameters
    input_data = np.array([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp_1]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = float(prediction[0][0])
    
    # Return result
    result = "Malignant" if probability > 0.5 else "Benign"
    return {
        "prediction": result, 
        "probability": f"{probability:.4f}",
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
    }

# Create Gradio interface first
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="BMI"),
        gr.Number(label="Glucose"),
        gr.Number(label="Insulin"),
        gr.Number(label="HOMA"),
        gr.Number(label="Leptin"),
        gr.Number(label="Adiponectin"),
        gr.Number(label="Resistin"),
        gr.Number(label="MCP-1")
    ],
    outputs=gr.JSON(label="Prediction Result"),
    title="Breast Cancer Prediction API",
    description="Enter patient data to predict breast cancer risk based on the Coimbra dataset.",
    article="""
    # Research Model - Breast Cancer Prediction
    
    This model is associated with academic research. If using this model in your work, 
    please cite the original research.
    
    ## Input Parameters:
    - **Age**: Patient's age in years
    - **BMI**: Body Mass Index
    - **Glucose**: Glucose level (mg/dL)
    - **Insulin**: Insulin level (µU/mL)
    - **HOMA**: Homeostatic Model Assessment
    - **Leptin**: Leptin level (ng/mL)
    - **Adiponectin**: Adiponectin level (µg/mL)
    - **Resistin**: Resistin level (ng/mL)
    - **MCP-1**: Monocyte Chemoattractant Protein 1 level (pg/dL)
    
    ## API Usage
    This interface can be used both interactively and programmatically via API calls.
    
    ### Direct API Endpoint
    For direct API access, use the following endpoint:
    `/predict`
    
    Example POST request:
    ```json
    {
      "data": [45.0, 21.3, 102.0, 2.0, 0.5, 22.7, 9.8, 7.2, 452.3]
    }
    ```
    """,
    examples=[
        [45.0, 21.3, 102.0, 2.0, 0.5, 22.7, 9.8, 7.2, 452.3],
        [57.0, 23.1, 97.0, 3.2, 0.7, 15.8, 12.3, 9.7, 534.6],
        [49.0, 26.8, 103.0, 4.1, 1.04, 18.2, 11.5, 8.1, 413.7]
    ]
)

# Create FastAPI app with an API endpoint matching Gradio's standard prediction path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add predict endpoint that matches what FlutterFlow expects
@app.post("/predict")
async def api_predict(request: Request):
    try:
        data = await request.json()
        inputs = data.get("data", [])
        
        if len(inputs) != 9:
            return JSONResponse(status_code=400, content={"error": "Expected 9 input parameters"})
        
        inputs = [float(x) for x in inputs]
        result = predict(*inputs)
        
        return JSONResponse(content={"data": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Also add the same endpoint at a path that's definitely going to work with HF Spaces
@app.post("/api/predict")
async def api_predict_alt(request: Request):
    try:
        data = await request.json()
        inputs = data.get("data", [])
        
        if len(inputs) != 9:
            return JSONResponse(status_code=400, content={"error": "Expected 9 input parameters"})
        
        inputs = [float(x) for x in inputs]
        result = predict(*inputs)
        
        return JSONResponse(content={"data": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Mount the Gradio app to the FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")

# For Hugging Face Spaces compatibility
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
