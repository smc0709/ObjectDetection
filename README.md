# Object Detection and Tracking
C Project that allows detecting and tracking objects moving in a sequence of frames using perceptual relevance metrics. 

## How to install
Download the repository. To create the executable file, run the next commands:
```
cd ObjectDetection
cmake .
make
```

Then create a new folder named "output" to save the resulting processed frames:
```
mkdir output
```

## How to use
To run the program use the following syntax:
```
./objectdetection [path/name] [extension of images] [numb of images to process]
```
Where the `[path/name]` means the path to the files and the part of the name of the files that remains constant between them all (the name without the frame number and without the file extension).

##### Example
Lets supose you want to process 600 frames saved inside a folder called `input` (inside the project), and all the frames are named like `frame1.bmp`, `frame2.bmp`, `frame3.bmp`, etc. Then you would do:
```
./objectdetection ./input/frame .bmp 600
```
The frames in the output will appear in the folder `./output` and look like `output1.bmp`, `output2.bmp`, `output3.bmp`, etc.

##### Notes
 - Do not use leading zeros in the frame numbering.
 - If you have a video, you can split it into frames using the ffmpeg free tool with the following command:
```
	ffmpeg -i "./inputVideo.mp4" "./ObjectDetection/input/frame%d.bmp" -hide_banner
```

## Dependencies
- CMake
- Make
- C compliler (gcc)
- It uses a library as git submodule: https://github.com/m-alarcon/PRMovement
