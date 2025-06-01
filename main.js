// =================== HTML Elements ===================
const body = document.querySelector("body");
const enableBtn = document.getElementById('enable');
const disableBtn = document.getElementById('disable');
const captureBtn = document.getElementById('capture');
const downloadBtn = document.getElementById('download');
const videoElement = document.querySelector('.input_video');
const canvasElement = document.querySelector('.output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const displayMode = document.querySelector(".displayMode");
const gestureText = document.querySelector(".gestureText");

const FRAME_LIMIT = 30; // Number of frames per gesture sample

// =================== MediaPipe Hands Setup ===================
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5,
});

// =================== Camera Setup ===================
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 640,
  height: 480
});

function enableCam() {
    if (camera.g) return;
    
    try {
        camera.start();
        console.log('Webcam started.');
        displayMode.textContent = 'MODE: PREDICT';
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

function disableCam() {
    if (!camera.g) return;
    
    try {
        camera.stop();
        console.log('Webcam stopped.');
        displayMode.textContent = 'MODE: NONE';
        enableBtn.disabled = false;
        disableBtn.disabled = true;
        captureBtn.disabled = true;
        downloadBtn.disabled = true;

    } catch (error) {
        console.error('Failed to stop webcam:', error);
        displayModeElement.textContent = `MODE: ERROR`;
        gestureTextElement.textContent = `GESTURE: Error: ${error.message}`;
        enableBtn.disabled = true;
        disableBtn.disabled = false;
        captureBtn.disabled = false;
        downloadBtn.disabled = false;
    }
}

// =================== Data Collection ===================
let modeRecord = false;
let modePredict = true;
let RecordSequence = [];
let collectedData = [];

function captureFrame(landmarks) {
  if (!landmarks) return;
  const frame = landmarks.flatMap(pt => [pt.x, pt.y, pt.z]);
  RecordSequence.push(frame);

  if (RecordSequence.length === FRAME_LIMIT) {
    const label = prompt("Enter a label for this gesture:").toUpperCase();
    collectedData.push({ label, sequence: [...RecordSequence] });
    console.log(`Saved sequence for label: ${label}`);
    RecordSequence = [];
  }
}

function toggleCaptureMode() {
  if (!modeRecord) {
    if (camera.i === 0) {
      alert("Turn on the camera first!");
      return;
    }
    modeRecord = true;
    modePredict = false;
    displayMode.textContent = "Mode: RECORD";
  } else {
    modeRecord = false;
    modePredict = true;
    displayMode.textContent = "Mode: PREDICT";
  }
}

function downloadDataset() {
  if (collectedData.length === 0) {
    alert("Dataset is empty!");
    return;
  }
  const blob = new Blob([JSON.stringify(collectedData)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "gesture_dataset.json";
  a.click();
  console.log("Dataset downloaded successfully.");
}

// =================== Loading labels ===================
fetch('./labels.json')
  .then(response => response.json())
  .then(data => labels = data)
  .catch(err => console.error("Failed to load labels:", err));

// =================== Prediction Setup ===================
let model = null;
let PredictSequence = [];
let predictedGesture = '';
let labels = null; // Customize for your labels
let MODEL_URL = "./model_tfjs/model.json"; // Path to your model

async function loadModel() {
  try {
    model = await tf.loadLayersModel(MODEL_URL);
    console.log("Model loaded!");
  } catch (err) {
    console.error("Model failed to load:", err);
  }
}

async function predictGesture(sequence) {
  if (!model || sequence.length !== FRAME_LIMIT || sequence[0].length !== 63) return;

  try {
    const inputTensor = tf.tensor(sequence).reshape([1, FRAME_LIMIT, 63]);
    const prediction = model.predict(inputTensor);

    // Checking the softmax probabilities
    // const predictionData = await prediction.data();
    // console.log("Prediction data:", Array.from(predictionData));

    const predictedIndex = (await prediction.argMax(-1).data())[0];

    const newGesture = labels[predictedIndex];
    if (newGesture !== predictedGesture) {
      predictedGesture = newGesture;
      console.log("Predicted:", predictedGesture);
    }

    tf.dispose([inputTensor, prediction]);
  } catch (err) {
    console.error("Prediction failed:", err);
  }
}

function addLandmarkFrame(landmarks) {
  if (!landmarks) return;
  const points = landmarks.flatMap(pt => [pt.x, pt.y, pt.z]);
  PredictSequence.push(points);

  if (PredictSequence.length > FRAME_LIMIT) {
    PredictSequence.shift();
  }

  if (PredictSequence.length === FRAME_LIMIT) {
    predictGesture(PredictSequence);
  }
}

function updateGestureText() {
  gestureText.textContent = `GESTURE: ${predictedGesture}`;
}

// =================== MediaPipe Callback ===================
hands.onResults((results) => {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
      drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });
    }

    const landmarks = results.multiHandLandmarks[0];
    if (modeRecord) {
      captureFrame(landmarks);
      PredictSequence = [];
    } else if (modePredict) {
      addLandmarkFrame(landmarks);
      updateGestureText();
    }
  }

  canvasCtx.restore();
});

// =================== Keyboard Controls ===================
body.addEventListener("keypress", (e) => {
  switch (e.key) {
    case 'e': if (!enableBtn.disabled) enableCam(); break;
    case 'd': if (!disableBtn.disabled) disableCam(); break;
    case 'r': if (!captureBtn.disabled) toggleCaptureMode(); break;
    case 's': if (!downloadBtn.disabled) downloadDataset(); break;
  }
});

// =================== Initialization ===================
window.onload = () => {
    loadModel();
    displayMode.textContent = "Mode: NONE";
    disableBtn.disabled = true;
    captureBtn.disabled = true;
    downloadBtn.disabled = true;
};
