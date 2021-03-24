# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report



### Reflection

### 1. Image Pipeline


The pipeline consists of 5 steps in total

1.) Convert to greyscale
2.) Image blur
3.) Canny edge detection
4.) Roi to filter elements on road
5.) Draw lines on image using hough_lines


Parameters for the pipeline are:


##### parameters

gaussian_blur_blur = 5        # parameter for grayscale(...)
canny_low_threshold = 150    # parameter for canny(...)
canny_high_threshold = 200   # parameter for canny(...)


parameters for hugh:

hugh_rho   = 1
hugh_theta   = np.pi/180
hugh_threshold   = 1
hugh_min_line_len   = 12
hugh_max_line_gap   = 2


The result can be seen here

The draw lines has been modified to seperate between left and right lines by calculation the slope.

By extracting the constant term from all lines simple averaging is possible.

### Improvements made for solidYellowLeft.mp4
To have more stable lines a little buffer was implemented to store the lines of the last 16 video frames.
This allows average and reduces jitter.



### 2. Shortcomings
(-) Different light conditions causes pipeline to fail
(-) Cars changing in front of cars are not tested
(-) low light is not tested (more noise in video)
(-) missing lines will cause pipeline to fail


### 3. Suggest possible improvements to your pipeline

Filter out lines that from other cars, do not belong to the street.
Weight lines by there lenght (we should trust longer detected lines more that short lines)
