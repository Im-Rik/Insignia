# Insignia: A Sign Language Recognition System

"Insignia" [Indian Sign-Language Interpretation Application] is a communication system designed to break down the barriers between the deaf and hard-of-hearing community and the hearing world. The project's ultimate vision is to create a seamless, real-time translator that can convert sign language gestures into text and then into audible speech.

Our journey began with this broad goal, and our first major step has been to build and validate the core technology within a critical and high-impact domain: healthcare.

## About The Project: Insignia for Healthcare

The current version of "Insignia" is an innovative sign language interpretation system tailored for medical environments. It addresses the urgent communication gap that often exists between deaf or nonverbal patients and healthcare professionals.

The system empowers patients to communicate their symptoms using only a standard webcam. The patient can perform a sign, and the system translates it into a complete, grammatically correct English sentence on a screen for the doctor to read instantly.

### Key Features

* **Accessible Vision-Based System:** It works with standard webcams, eliminating the need for expensive or specialized hardware like sensor gloves or depth cameras.
* **Healthcare-Focused Vocabulary:** The model is trained on 24 distinct Indian Sign Language (ISL) words carefully selected for their relevance in a medical context, such as "Doctor," "Medicine," "sick," and "Family".
* **Enhanced Communication with Contextual Cues:** The system allows patients to select supplementary emojis or images (e.g., icons for pain levels or fever) to add crucial context and emotion to their message, enabling more nuanced communication.
* **Intelligent Sentence Generation:** Insignia uses a powerful Large Language Model (LLM) to synthesize recognized signs and contextual cues into coherent, grammatically correct sentences, providing clear and actionable communication for medical personnel.
* **Proven Accuracy:** The core recognition model achieved a 92.14% accuracy on the held-out test dataset, demonstrating its reliability.

### How It Works: Technology Stack

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

The core of the project is a lightweight, attention-based Bidirectional Long Short-Term Memory (Bi-LSTM) network built in PyTorch. This model is specifically designed to process the temporal dependencies inherent in sign language gestures. An integrated attention mechanism allows the model to dynamically weigh the importance of different segments of a sign, improving focus and accuracy.

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
git clone https://github.com/Im-Rik/Insignia
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
