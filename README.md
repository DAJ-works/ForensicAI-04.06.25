# ForensicAI

#### Developers: Dhruv Kothari, Aarav Goel, Jayanth Veerappa
#### Los Altos Hacks 2025 - Winners: 1st Place

CCTV Insight Analyzer is a powerful application designed to help users extract actionable intelligence from CCTV footage. By combining computer vision and large language models (LLMs), this tool automatically analyzes video input and presents a comprehensive breakdown of events, individuals, and objects, all through a clean and interactive interface.

## What It Does
Simply upload a CCTV video, and the application will:

**Identify Every Person**: Detects and lists all individuals who appear in the footage, with timestamps of their appearances.

**Recognize Objects and Items**: Automatically detects and catalogs items and objects present throughout the footage.

**Generate a Timeline of Events**: Creates a detailed, scrollable timeline summarizing what happened, when it happened, and who was involved.

**Chat with the Case AI**: Users can interact with a large language model to ask questions about the case, clarify details, or explore hypotheses using natural language.

## Features
### Feature	Description
**CCTV Upload**	Upload surveillance footage securely and privately.
**AI-Powered Analysis**	Uses deep learning to detect faces, track movement, and classify objects.
**Interactive Analytics Page**	Visualizes people, items, and timeline in a user-friendly dashboard.
**Evidence Logging**	Items and people are automatically logged and time-tagged.
**LLM Chat Assistant**	Chat interface to ask the AI questions about the footage or events.
**Secure & Private**	Keeps footage confidential and processed with privacy in mind.
# Use Cases
**Crime Investigation** – Review suspects and item movements in real time.

**Retail Loss Prevention** – Track missing items or suspicious behaviors.

**Security Monitoring** – Understand incidents faster with a visual timeline.

**Legal Documentation** – Export structured data and transcripts for legal use.

# Tech Stack
**Frontend**: React, CSS

**Backend**: Python (FastAPI), Node.js

**Video Processing**: OpenCV, YOLOv8, PyTorch

**LLM Integration**: HuggingFace


# Getting Started
**Clone the repository**

```git clone https://github.com/your-username/cctv-insight-analyzer.git
cd cctv-insight-analyzer
Install dependencies

 Backend
cd backend
pip install -r requirements.txt

 Frontend
cd ../frontend
npm install
Start the application

 Run backend
cd backend
uvicorn main:app --reload

 Run frontend
cd ../frontend
npm start
```
## Use Cases ##
Crime Investigation – Review suspects and item movements in real time.
Retail Loss Prevention – Track missing items or suspicious behaviors.
Security Monitoring – Understand incidents faster with a visual timeline.
Legal Documentation – Export structured data and transcripts for legal use.

## RAG Implementation for LLM Chat Interface ##

Our RAG system processes video analysis results to enable context-aware conversations:

- Video analysis results (people, objects, timeline events) are structured into specialized document formats
- Each document includes metadata and context for efficient retrieval
- Analysis results are chunked into semantically meaningful segments

- User queries are analyzed for intent recognition
- A specialized prompt template incorporates retrieved context
- Multi-stage retrieval ensures the most relevant information is provided

## Custom YOLOv8 Weapon Detection Model ##

- We trained a specialized YOLOv8 model for weapon detection
- Training Methodology: Trained for 50 epochs using google collab GPU

## Future Work ###
- Integration with live video feeds
- Multi-camera synchronization and analysis
- Behavioral anomaly detection
- Mobile application for field investigations
- Enhanced visualization tools for event reconstruction
