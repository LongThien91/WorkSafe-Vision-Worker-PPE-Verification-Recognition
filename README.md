# WorkSafe-Vision-Worker-PPE-Verification-Recognition  
 Project uses video or image streams to monitor workers’ compliance with personal protective equipment (PPE) requirements. The system applies instance segmentation to detect individual workers, checks for safety vests and hard hats, and if non-compliance is detected, performs facial recognition to identify the worker.
## How to run
### Clone the repository
```bash
git clone https://github.com/LongThien91/WorkSafe-Vision-Worker-PPE-Verification-Recognition.git
```
### Install requirement
```bash
pip install -r requirements.txt
```
### run python file
```bash
py WorkSafeVision.py
```
1. Select Detection Mode:  
Press 0 to use your camera for real-time object detection.  
Press 1 to use a video file for offline analysis.  

2. Optional: Face Recognition Mode  
You can choose to enable face recognition mode for both camera and video detection. You can replace all the picture of worker in WorkerFaceImage folder.

3. Video Path (if selected in step 1):  
Provide the path to your video file.  
Example:  
```bash
Enter your video path here:D:\DownLoad\VideoDownload\ppe-1.mp4
```


