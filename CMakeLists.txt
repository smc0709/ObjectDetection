cmake_minimum_required(VERSION 2.8)
project( ObjectDetection )
add_executable( objectdetection pr_object_detection.c ./PRMovement/pr_movement.c ./PRMovement/bmpreader.c ./PRMovement/yuv_rgb.c ./PRMovement/imgutils.c)
target_link_libraries( objectdetection -lm )
