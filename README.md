## Prerequisites

Before you run this project, make sure the following are installed on your machine:

- [Python](https://www.python.org/downloads/)
- [Node.js](https://nodejs.org/)
- [FFmpeg](https://ffmpeg.org/download.html)

---

## Set Up Groq API Key

To use the Groq API:

1. Go to [GroqCloud Console](https://console.groq.com/)
2. Sign up for a free account (as of 17-06-25) if you don’t have one.
3. Navigate to **API Keys** and create a new **secret key**.
4. Copy the key and set it as an environment variable.

### Set Environment Variable (Windows)

Open Command Prompt and run:

```bash
setx GROQ_API_KEY "your_api_key_here"
```

**Example:**

```bash
setx GROQ_API_KEY "gsk_7dS4p8XLbYV4nlwWGdyb3FYatNlsQvOQDYPHRTccePax"
```

> ℹ Note: This sets the variable permanently for your user profile.

---

## Getting Started

### 1. Clone or Download the Repository

Make sure all project files are downloaded or cloned to your machine.

---

### 2. Start Backend Services

Navigate to the backend folder:

```bash
cd Insignia/backend/
```

Run all backend servers:

```bash
python run_all_servers.py
```

---

### 3. Start Frontend Development Server

In a new terminal:

```bash
cd Insignia/frontend/
npm install
npm run dev
```

This installs dependencies and launches the frontend server.

---

### 4. Open the Application

In your browser (preferably **Firefox**), go to:

```
http://localhost:5173/
```

You're good to go! 

---

## Removing the Environment Variable (Optional)

To delete the `GROQ_API_KEY` environment variable later:

Open Command Prompt and run:

```bash
REG DELETE HKCU\Environment /F /V GROQ_API_KEY
```

---

Let me know if you want this formatted as a `README.md` file with markdown badges or collapsible sections.
