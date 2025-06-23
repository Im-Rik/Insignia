# Insignia: A Sign Language Recognition System

"Insignia" [Indian Sign-Language Interpretation Application] is a communication system designed to break down the barriers between the deaf and hard-of-hearing community and the hearing world. The project's ultimate vision is to create a seamless, real-time translator that can convert sign language gestures into text and then into audible speech.

Our journey began with this broad goal, and our first major step has been to build and validate the core technology within a critical and high-impact domain: healthcare.

## About The Project: Insignia for Healthcare

The current version of "Insignia" is an innovative sign language interpretation system tailored for medical environments. It addresses the urgent communication gap that often exists between deaf or nonverbal patients and healthcare professionals.

The system empowers patients to communicate their symptoms using only a standard webcam. The patient can perform a sign, and the system translates it into a complete, grammatically correct English sentence on a screen for the doctor to read instantly.

## Application Modes and Functionality

The application is designed with several distinct modes, each catering to a different use case and handled by a specific backend server.

### Upload Mode (Single Gesture Recognition)

* **Purpose:** Designed for recognizing a single sign from a short, pre-recorded video.
* **Workflow:** The user uploads a video file containing one gesture. The system processes the video and returns the classification for that sign.
* **Server:** This functionality is handled by `server_1`.

### Live / Dev Mode (Real-time Feed Recognition)

* **Purpose:** Provides real-time sign language prediction from a live webcam feed.
* **Workflow:** This mode uses a sliding window approach to continuously analyze the video stream and predict gestures as they are being performed.
* **Versions:**
    * **Live:** A simplified, lightweight version for quick, real-time predictions.
    * **Dev:** A more computationally intensive version that provides additional diagnostics, such as displaying the MediaPipe landmark overlay and other statistics.
* **Server:** Both Live and Dev modes are handled by `server_2`.

### Dev2 Mode (Manual Long Video Segmentation)

* **Purpose:** Allows for the analysis of long videos containing multiple signs.
* **Workflow:** The user can upload or record a long video. Using built-in editor tools, the user manually marks the start and end points of each individual gesture within the video. These segmented clips are then sent to the model for classification.
* **Server:** This mode is handled by `server_3`.

### Dev3 Mode (Automated Segmentation and Sentence Generation)

* **Purpose:** This is the most advanced mode, automating the process of recognizing multiple signs from a long video and forming a complete sentence.
* **Workflow:**
    1.  **Automated Segmentation:** The system automatically segments a long video into individual sign clips using a Lucas-Kanade optical flow approach. This technique detects motion to identify when a sign begins and ends.
    2.  **Contextual Cues:** After segmentation, the user can select supplementary contextual cues (emojis or images) to accompany the sequence of recognized signs.
    3.  **LLM Sentence Generation:** The sequence of recognized gloss words and the selected contextual hints are sent to a Large Language Model (LLM). The LLM synthesizes these inputs to generate a grammatically correct and semantically coherent English sentence.
* **Implementation Note:** The integration of contextual cues and the LLM is currently implemented exclusively in the `dev3` mode.
* **Servers:** The segmentation process is handled by a `helper_server`, while the classification and LLM integration are managed by `server_3`.

Youtube Link - https://youtu.be/EmS7cnPimVg

## How It Works: Technology Stack

"Insignia" is a full-stack application built with a modern technology stack to ensure performance, reliability, and a smooth user experience.

#### **Frontend (User Interface)**

* **React:** Utilized for constructing a dynamic and responsive user interface.
* **Socket.IO Client:** Facilitates real-time, bidirectional communication with the backend to provide users with live status updates during processing.
* **Tailwind CSS:** Employed for streamlined styling of the user interface.

#### **Backend (Processing Engine)**

* **Flask:** A lightweight Python micro web framework that serves as the foundation of the backend application.
* **PyTorch:** The deep learning framework powering the `SignRecognizer` model, handling the complex neural network computations.
* **MediaPipe:** A framework from Google used for the crucial first step of extracting detailed pose, face, and hand landmarks from video frames. The system extracts a 1662-dimensional feature vector for each frame.
* **Flask-SocketIO:** Provides robust WebSocket integration for the Flask application, supporting communication with the frontend.
* **FFMPEG:** A powerful command-line tool used for handling video processing and conversion.
* **APScheduler:** A scheduling library used for background tasks, such as automated resource management and cleanup on the server.

### Model Architecture

The core of the project is a lightweight, attention-based Bidirectional Long Short-Term Memory (Bi-LSTM) network built in PyTorch. This model is specifically designed to process the temporal dependencies inherent in sign language gestures. An integrated attention mechanism allows the model to dynamically weigh the importance of different segments of a sign, improving focus and accuracy. The model achieved a 92.14% test accuracy on a curated dataset of 24 Indian Sign Language medical and relational signs.

---

## Getting Started

### Prerequisites

Before you run this project, make sure the following are installed on your machine:

-   [Python](https://www.python.org/downloads/)
-   [Node.js](https://nodejs.org/)
-   [FFmpeg](https://ffmpeg.org/download.html)

### Setup and Installation

#### 1. Clone the Repository

First, clone or download the repository to your local machine.

```bash
git clone <repository-url>
cd Insignia
```

#### 2. Set Up Groq API Key

The application uses the Groq API for Large Language Model integration.

1.  Go to the [GroqCloud Console](https://console.groq.com/).
2.  Sign up for a free account if you do not have one.
3.  Navigate to the **API Keys** section and create a new **secret key**.
4.  Copy the key and set it as an environment variable on your system.

**For Windows:**
Open Command Prompt and run the following command, replacing `"your_api_key_here"` with the key you copied.

```bash
setx GROQ_API_KEY "your_api_key_here"
```

This sets the environment variable permanently for your user profile. You may need to restart your terminal for the change to take effect.

#### 3. Start Backend Services

Navigate to the backend directory and run the server script.

```bash
cd backend/
python run_all_servers.py
```

This will start all the necessary backend processes.

#### 4. Start Frontend Development Server

Open a new terminal, navigate to the frontend directory, install dependencies, and start the development server.

```bash
cd ../frontend/
npm install
npm run dev
```

#### 5. Open the Application

Once the servers are running, open your web browser (preferably **Firefox**) and navigate to:

```
http://localhost:5173/
```

The application should now be running.

---

## Future Roadmap

The current healthcare application is the foundation. Our future work is focused on expanding "Insignia" to achieve our original vision of a universal sign language translator.

* **Gesture -> Text -> Speech:** The next major milestone is to add a Text-to-Speech (TTS) module. This will complete the translation pipeline, allowing the system to not only display the translated text but also speak it aloud for a more natural, conversational flow.
* **Vocabulary and Language Expansion:** We will progressively expand the vocabulary beyond the initial medical dataset to cover a wider range of daily communication topics and incorporate other sign languages.
* **Continuous Sign Language Recognition:** We plan to move from recognizing isolated signs to interpreting continuous, flowing sign language as it is used in natural conversation.
* **Enhanced Robustness and Personalization:** Future iterations will focus on improving model performance in challenging real-world conditions (like varying lighting or cluttered backgrounds) and allowing it to adapt to the unique signing styles of individual users.

## Removing the Environment Variable (Optional)

If you need to remove the `GROQ_API_KEY` environment variable from your system later, you can do so with the following command in Command Prompt:

```bash
REG DELETE HKCU\Environment /F /V GROQ_API_KEY
```

---

## Links

* **Youtube** - https://youtu.be/EmS7cnPimVg
* **Dataset** - https://www.kaggle.com/datasets/linardur/include-24-medical-modified/data
* **Model Details** - https://github.com/exotic123567/Insignia
