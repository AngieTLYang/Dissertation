Image Processing Pipeline with Socket Server

This project implements a server-side image processing pipeline that receives images from a client device (e.g., phone), detects pens in the image, extracts regions of interest, and performs OCR + LLM-based question answering. Communication between the client and server is managed via TCP sockets for both image streaming and control messages.

Image processing pipeline:
1. (owlv2_singleImage.py) Pen detection: Uses OWLv2 (and optionally YOLO) to detect pens in images.
2. (doclayout_singleImage.py) Document layout analysis: Extracts layout regions using DocLayout.
3. (filterByVisualCue.py) Visual cue filtering: Identifies text regions bounded by pens.
4. (tesseractAndGemini.py & test_callGemini.py) OCR & LLM integration: Performs text extraction (Tesseract) and generates answers using Gemini.

To let the server accept connections on the required ports, you need to allow inbound TCP connections. Run the following commands in an Administrator Command Prompt:

```cmd
netsh advfirewall firewall add rule name="Allow Command Port 12345" dir=in action=allow protocol=TCP localport=12345
netsh advfirewall firewall add rule name="Allow Command Port 12346" dir=in action=allow protocol=TCP localport=12346

