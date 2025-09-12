Image Processing Pipeline with Socket Server

This project implements a server-side image processing pipeline that receives images from a client device (e.g., phone), detects pens in the image, extracts regions of interest, and performs OCR + LLM-based question answering. Communication between the client and server is managed via TCP sockets for both image streaming and control messages.

(Server/serverForA32WithControl.py) Server-side Control Logic & Synchronization:
The system synchronizes image processing and client communication with two key events:
1. new_image_event -> signals when a fresh image arrives.
2. pause_event -> controls whether the processing loop is active or paused.

*Image Arrival
The phone client sends an image, the server saves it, and then sets the new_image_event.

*Processing Loop
Waits for new_image_event.
Also checks pause_event. If paused, it stops until resumed.
Otherwise, carry on the Image processing pipeline (below)

*Pause/Resume Mechanism
If exactly 2 pens are detected ->
The server sends PAUSE to the client.
Clears pause_event -> halts new processing.
Processing resumes only when a control client sends RESUME -> sets pause_event.

(image_processing/*) Image processing pipeline:
1. (owlv2_singleImage.py) Pen detection: Uses OWLv2 (and optionally YOLO) to detect pens in images.
2. (doclayout_singleImage.py) Document layout analysis: Extracts layout regions using DocLayout.
3. (filterByVisualCue.py) Visual cue filtering: Identifies text regions bounded by pens.
4. (tesseractAndGemini.py & test_callGemini.py) OCR & LLM integration: Performs text extraction (Tesseract) and generates answers using Gemini.

To enable the server to accept connections on the required ports, you must allow inbound TCP connections. Run the following commands in an Administrator Command Prompt:

```cmd
netsh advfirewall firewall add rule name="Allow Command Port 12345" dir=in action=allow protocol=TCP localport=12345
netsh advfirewall firewall add rule name="Allow Command Port 12346" dir=in action=allow protocol=TCP localport=12346

