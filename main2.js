// --- Get references to HTML elements ---
const videoElement = document.querySelector('.input_video');
// const canvasElement = document.querySelector('.output_canvas');
// const canvasCtx = canvasElement.getContext('2d');
const displayModeElement = document.querySelector('.displayMode');
const gestureTextElement = document.querySelector('.gestureText');

const enableBtn = document.getElementById('enable');
const disableBtn = document.getElementById('disable');
const captureBtn = document.getElementById('capture');
const downloadBtn = document.getElementById('download');

const hiddenCanvasElement = document.createElement('canvas'); // Create dynamically
hiddenCanvasElement.width = 224; // Model expects 224x224
hiddenCanvasElement.height = 224;
const hiddenCanvasCtx = hiddenCanvasElement.getContext('2d');

let model;
const MODEL_PATH = './model_tfjs2/model.json'; 

// --- 1. Load the model ---
async function loadMyModel() {
    try {
        // Use tf.loadLayersModel for Keras models saved with model.save()
        // Use tf.loadGraphModel for frozen graphs or models saved with tf.saved_model.save()
        model = await tf.loadLayersModel(MODEL_PATH); 
        // model.summary(); // To log the model architecture
        console.log('Model loaded successfully!');
        displayModeElement.textContent = 'MODE: IDLE';
        gestureTextElement.textContent = 'GESTURE: Waiting for hand...';
    } catch (error) {
        console.error('Failed to load model:', error);
        displayModeElement.textContent = `MODE: ERROR`;
        gestureTextElement.textContent = `GESTURE: Error: ${error.message}`;
        // Disable prediction related functionality if model fails to load
        // You might want to disable capture/download buttons too
        enableBtn.disabled = true; 
    }
}

// --- 2. Initialize the MediaPipe Hands ---
const hands = new Hands({
    locateFile: (file) => {
        // This tells MediaPipe where to find its WASM and other model assets.
        // It's crucial for MediaPipe to work correctly.
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 2, // Detect up to 1 hand. Adjust to 2 if your model handles two hands.
    modelComplexity: 1, // Use 0 for a lighter model, 1 for a more complex one. If not mentioned it might go to 2 which is a more complex model.
    // The modelComplexity setting is crucial. If your model was trained on a specific complexity.
    minDetectionConfidence: 0.7, // Minimum confidence for hand detection
    minTrackingConfidence: 0.5 // Minimum confidence for hand tracking across frames
});

// --- 3. Define MediaPipe's onResults callback ---
// This function is called every time MediaPipe processes a video frame
// and has hand landmark detection results.
hands.onResults((results) => {
    // Clear the canvas for drawing new results
    // canvasCtx.save();
    // canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height); // Draw the camera feed onto the canvas (mirrored horizontally)
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0 && model) {
        // --- Process the FIRST detected hand for image input ---
        const firstHandLandmarks = results.multiHandLandmarks[0];

        // 1. Calculate a bounding box from the landmarks
        let minX = 1, minY = 1, maxX = 0, maxY = 0;
        for (const landmark of firstHandLandmarks) {
            minX = Math.min(minX, landmark.x);
            minY = Math.min(minY, landmark.y);
            maxX = Math.max(maxX, landmark.x);
            maxY = Math.max(maxY, landmark.y);
        }

        // Convert normalized coordinates to pixel coordinates on the *original* results.image
        const imgWidth = results.image.width;
        const imgHeight = results.image.height;

        const paddingRatio = 0.2; 
        let cropX = minX * imgWidth;
        let cropY = minY * imgHeight;
        let cropWidth = (maxX - minX) * imgWidth;
        let cropHeight = (maxY - minY) * imgHeight;

        // Apply padding
        cropX = Math.max(0, cropX - cropWidth * paddingRatio / 2);
        cropY = Math.max(0, cropY - cropHeight * paddingRatio / 2);
        cropWidth = Math.min(imgWidth - cropX, cropWidth * (1 + paddingRatio));
        cropHeight = Math.min(imgHeight - cropY, cropHeight * (1 + paddingRatio));

        if (cropWidth <= 0 || cropHeight <= 0) {
            console.warn("Invalid hand bounding box dimensions. Skipping prediction.");
            gestureTextElement.textContent = 'GESTURE: Hand detected but cannot crop.';
            // canvasCtx.restore(); // REMOVE THIS
            return;
        }

        // --- No drawing on main canvas for bounding box, as main canvas is removed ---

        // 2. Clear and draw the cropped & resized image onto the hidden canvas
        hiddenCanvasCtx.clearRect(0, 0, hiddenCanvasElement.width, hiddenCanvasElement.height);
        hiddenCanvasCtx.drawImage(
            results.image, 
            cropX, cropY, cropWidth, cropHeight, 
            0, 0, hiddenCanvasElement.width, hiddenCanvasElement.height 
        );

        // 3. Convert hidden canvas pixels to a TensorFlow.js tensor
        tf.tidy(() => {
            const inputTensor = tf.browser.fromPixels(hiddenCanvasElement); 
            const expandedTensor = inputTensor.expandDims(0); 
            const normalizedTensor = expandedTensor.div(255.0); 

            // 4. Make prediction with your model
            const prediction = model.predict(normalizedTensor);
            
            const predictedClassIndex = prediction.argMax(1).dataSync()[0];
            const confidenceScores = prediction.dataSync(); 
            const confidence = confidenceScores[predictedClassIndex] * 100; 
            
            const classLabels = ['Hello', 'Yes', 'No', 'Thank You', 'I Love You', 'Other/None', /* add your other 7 labels here to make 13 */ ]; 
            // IMPORTANT: Ensure your classLabels array has 13 entries matching your model's output order!
            
            gestureTextElement.textContent = 
                `GESTURE: ${classLabels[predictedClassIndex]} (${confidence.toFixed(2)}%)`;
        });

        // --- No drawing of MediaPipe landmarks, as main canvas is removed ---

    } else {
        // No hands detected or model not loaded
        if (model) {
            gestureTextElement.textContent = 'GESTURE: No hand detected.';
        } else {
            gestureTextElement.textContent = 'GESTURE: Model not loaded yet.';
        }
    }

    // canvasCtx.restore(); // Restore canvas state after drawing
});

// --- Camera Setup ---
const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
        // else console.log("hands.send skipped: Model not loaded yet.");
    },
    width: 640,
    height: 480
});

// --- Webcam Control Functions ---

// Function to enable the webcam
async function enableCam() {
    if (camera.g) return; // If camera is already enabled, do nothing
    try {
        await camera.start();
        console.log('Webcam started.');
        displayModeElement.textContent = 'MODE: PREDICT';
        enableBtn.disabled = true;
        disableBtn.disabled = false;
        captureBtn.disabled = false;
        downloadBtn.disabled = false;
    } catch (error) {
        console.error('Failed to start webcam:', error);
        displayModeElement.textContent = `MODE: ERROR`;
        gestureTextElement.textContent = `GESTURE: Error: ${error.message}`;
        enableBtn.disabled = false;
        disableBtn.disabled = true;
        captureBtn.disabled = true;
        downloadBtn.disabled = true;
    }
}

// Function to disable the webcam
function disableCam() {
    if (!camera.g) return; // If camera is not enabled, do nothing

    camera.stop();
    console.log('Webcam stopped.');
    displayModeElement.textContent = 'MODE: IDLE';
    gestureTextElement.textContent = 'GESTURE: Webcam OFF';
    enableBtn.disabled = false;
    disableBtn.disabled = true;
    captureBtn.disabled = true;
    downloadBtn.disabled = true;
}

// --- Data Capture Functions (Conceptual) ---

// Placeholder for data capture logic
function ClickedCapture() {
    console.log("CAPTURE button clicked!");
    // You would implement logic here to:
    // 1. Get the current `featureVector` (from `hands.onResults`).
    // 2. Pair it with a label (e.g., from a dropdown or user input).
    // 3. Store it in an array or object in memory.
    // For dynamic gestures, you'd collect a sequence of featureVectors over time.
    alert("Capture functionality needs to be implemented. (Collecting data points)");
}

// Placeholder for downloading the dataset
function downloadDataset() {
    console.log("DOWNLOAD THE DATA button clicked!");
    // You would implement logic here to:
    // 1. Take the collected data from `ClickedCapture()`.
    // 2. Convert it into a downloadable format (e.g., JSON, CSV).
    // 3. Trigger a file download.
    alert("Download functionality needs to be implemented. (Saving collected data)");
}

// --- Initial Setup on Page Load ---
document.addEventListener('DOMContentLoaded', () => {
    loadMyModel(); // Load the TF.js model first

    // Initial button states
    enableBtn.disabled = false;
    disableBtn.disabled = true;
    captureBtn.disabled = true;
    downloadBtn.disabled = true;
});

// Set canvas dimensions to match video once video metadata loads
// (This handles cases where video might load before enableCam is clicked)
// videoElement.addEventListener('loadedmetadata', () => {
//     canvasElement.width = videoElement.videoWidth;
//     canvasElement.height = videoElement.videoHeight;
// });

// ====== Event Listeners for Button Clicks ==== 
document.addEventListener("keypress", (e) => {
  switch (e.key) {
    case 'e': if (!enableBtn.disabled) enableCam(); break;
    case 'd': if (!disableBtn.disabled) disableCam(); break;
    case 'r': if (!captureBtn.disabled) toggleCaptureMode(); break;
    case 's': if (!downloadBtn.disabled) downloadDataset(); break;
  }
});