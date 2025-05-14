# SeeWithMe

# Project Description
Ally is a real-time voice and vision assistant.  
It listens to user requests, responds naturally using voice, and activates the camera whenever visual understanding is required — such as detecting objects or interpreting a scene.

Built using:
- Deepgram STT + ElevenLabs TTS
- YOLOv8 + Gemini Vision for scene understanding
- GPT-4o for intelligent conversations
- LiveKit Agents for real-time media + audio routing


# Setup

First, create a virtual environment, update pip, and install the required packages:

```
$ python3 -m venv ally_env
$ source ally_env/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

You need to set up the following environment variables:

```
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
ELEVEN_API_KEY=...
GEMINI_API_KEY = ...
```

Then, run the assistant:

```
$ python3 assistant.py download-files
$ python3 assistant.py start

```

Finally, you can load the [hosted playground](https://agents-playground.livekit.io/) and connect it.

##  Detailed Flowchart Overview
![System Flowchart](FinalDesign.png)


