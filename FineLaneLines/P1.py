#!/usr/bin/env python
# coding: utf-8

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)



def draw_lines_v2(img, lines, color=[255, 0, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    

    global global_right_line_buffer_m
    global global_right_line_buffer_b
    global global_left_line_buffer_m
    global global_left_line_buffer_b


    color_green = [0, 255, 0]
    color_red = [255, 0, 0]

    b_left = []
    b_right = []


    m_left = []
    m_right = []

    # print ( len(lines) )

    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y2 - y1) /  (x2  - x1) # calculate slope of line
            
#            length = np.sqrt ( np.power(x2 - x1 , 2) +  np.power(y2 - y1, 2) )
            b = y1 - m * x1 # constant term of line as linear function
            
            if (m > 0): # store right lane values in list
                b_right.append(b)
                m_right.append(m)
            if (m < 0): # store left lane values in list
                b_left.append(b)
                m_left.append(m)
    # print (b_left, b_right)
    
    
    # store the average of the detected left lines in the line buffer
    if (len(b_left) != 0):
        global_left_line_buffer_m.append(median (m_left)) #(sum(m_left) / len(m_left))
        global_left_line_buffer_b.append(median (b_left)) #(sum(b_left) / len(b_left))

    # store the average of the detected right lines in the line buffer
    if (len(b_right) != 0):
        global_right_line_buffer_m.append(median(m_right)) #(sum(m_right) / len(m_right))
        global_right_line_buffer_b.append(median(b_right)) #(sum(b_right) / len(b_right))  


    # The buffer size is limited to global_line_buffer_max_elements (default = 8)
    # remove first element in list is buffer exceeds this value
    if (len (global_left_line_buffer_b) == global_line_buffer_max_elements):
        global_left_line_buffer_m.pop(0)
        global_left_line_buffer_b.pop(0)

    if (len (global_right_line_buffer_b) == global_line_buffer_max_elements):
        global_right_line_buffer_m.pop(0)
        global_right_line_buffer_b.pop(0)


    # only draw lines if there are elements in the buffer
    if (len (global_left_line_buffer_b) > 0):
        b_left_average =  median(global_left_line_buffer_b) #sum (global_left_line_buffer_b) / len (global_left_line_buffer_b)
        m_left_average =  median(global_left_line_buffer_m) #sum (global_left_line_buffer_m) / len (global_left_line_buffer_m)
        cv2.line(img, ( int((540 - b_left_average) / m_left_average) , 540), (int((320 - b_left_average) / m_left_average), 320), color_green, thickness)

    if (len (global_right_line_buffer_b) > 0):
        b_right_average = median(global_right_line_buffer_b) # sum (global_right_line_buffer_b) / len (global_right_line_buffer_b)
        m_right_average = median(global_right_line_buffer_m) # sum (global_right_line_buffer_m) / len (global_right_line_buffer_m)
        cv2.line(img, ( int((540 - b_right_average) / m_right_average) , 540), (int((320 - b_right_average) / m_right_average), 320), color_red, thickness)


def hough_lines_v2(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    draw_lines_v2(line_img, lines)
    return line_img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)



import os



images = os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[ ]:

# Building the pipeline

# Color-Image -> convert to gray -> blur -> cany detection -> Region of interest 


# parameters
gaussian_blur_blur = 5               # parameter for grayscale(...)
canny_low_threshold = 150    # parameter for canny(...)
canny_high_threshold = 200   # parameter for canny(...)

hugh_rho   = 1
hugh_theta   = np.pi/180
hugh_threshold   = 1
hugh_min_line_len   = 12
hugh_max_line_gap   = 2

imshape = image.shape
roi_vertices = [[(125,imshape[0]),(460, 300), (500, 300), (860,imshape[0])]]

for image_name in images:

    image = mpimg.imread("test_images/" + image_name)

    image_grayscale = grayscale(image)
    image_blur      = gaussian_blur(image_grayscale, gaussian_blur_blur)
    image_canny     = canny(image_blur, canny_low_threshold, canny_high_threshold)
    image_roi       = region_of_interest(image_canny, np.array( roi_vertices, dtype=np.int32 ) )
    image_hough     = hough_lines(image_roi, hugh_rho, hugh_theta, hugh_threshold, hugh_min_line_len, hugh_max_line_gap)

    image_weighted = weighted_img(image_hough, image)
    plt.imshow(image_weighted)
    #plt.show()


    #mpimg.imsave("test_images_output/" + 'w_' + image_name, image_weighted)



# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[ ]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
#from IPython.display import HTML


# In[ ]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    image_grayscale = grayscale(image)
    image_blur      = gaussian_blur(image_grayscale, gaussian_blur_blur)
    image_canny     = canny(image_blur, canny_low_threshold, canny_high_threshold)
    image_roi       = region_of_interest(image_canny, np.array( roi_vertices, dtype=np.int32 ) )
    image_hough     = hough_lines_v2(image_roi, hugh_rho, hugh_theta, hugh_threshold, hugh_min_line_len, hugh_max_line_gap)

    image_weighted = weighted_img(image_hough, image)


    return image_weighted


# Let's try the one with the solid white lane on the right first ...

# In[ ]:

# implementing a buffer to store the lines from previous image frames
# size of buffer is defined by global_line_max_elements
global_line_buffer_max_elements = 16

global_right_line_buffer_m = []  
global_right_line_buffer_b = []  

global_left_line_buffer_m = []
global_left_line_buffer_b = []




from statistics import median

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile("a:\\white.mp4", audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile("a:\\yellow.mp4", audio=False)

quit()

challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
#get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


