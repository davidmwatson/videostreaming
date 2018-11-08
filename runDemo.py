"""
Demo of running live video stream and displaying on Oculus via PsychoPy.
Also shows option of applying post-processing function to manipulate images
(in this case, a simple function for photo negating the images). Application
of post-processing function can be toggled on and off by pressing space bar.
Press escape or q to quit.

Script is currently configured for an IDS uEye 3220CP camera, which returns a
752x480 resolution monochrome feed.

Camera can run with either OpenCV or uEye-SDK backends.  OpenCV is more
general purpose and should work with pretty much any camera, but may not be
able to access manufacturer-specific camera functions (e.g. freerun modes for
achieving higher frame rates, automatic gain control, etc.). The uEye-SDK will
only work on IDS uEye cameras, but does allow access to functions specific
to those cameras.  It should be easy to write further backends for other
manufacturers' SDKs as necessary.

Oculus display uses psychxr library (https://github.com/mdcutone/psychxr) and
the visual.Rift class available in PsychoPy versions > 3.0.  Note that this
only works for 64bit Python3 installations running on Windows OS.  This means
this will NOT work for the PsychoPy standalone as this doesn't offer 64bit.
"""

from psychopy import visual, event

# local imports
from videostreaming import OpenCV_VideoStream, uEyeVideoStream, FrameStim


### Key variables - update as needed ###

# Proportion of Oculus FOV to use for displaying image.  The Oculus FOV is
# larger than my camera's FOV, so I've restricted the display a little so it
# doesn't appear overly magnified
mag_factor = 0.5

# Which backend to use - set to 'opencv' or 'ueye'
camera_backend = 'opencv'

# Camera settings for opencv backend
opencv_settings = {
        'cam_num':1, # should select 1st external camera
        'cam_res':(752,480),  # res for uEye 3220CP cam
        'fps':45,
        'colour_mode':'mono',
        'vertical_reverse':True  # uEye returns images upside-down by default!
        }

# Camera settings for uEye backend
ueye_settings = {
        'fps':90,
        'pixel_clock':'max',
        'colour_mode':'mono',
        'auto_exposure':'camera',
        'auto_gain_control':'camera'
        }


### Custom functions ###
def negate(frame):
    """Example post-processing function.  Photo negates image."""
    return 255 - frame


### Begin main script ###

# Open camera stream
if camera_backend == 'opencv':
    stream = OpenCV_VideoStream(**opencv_settings)
elif camera_backend == 'ueye':
    stream = uEyeVideoStream(**ueye_settings)
else:
    raise RuntimeError('Unknown backend')

# Set post-proc function, default application to False
stream.setPostproc(negate)
stream.switchApplyPostproc(False)

# Open handle to rift
hmd = visual.Rift(monoscopic=True, color=-1,  warnAppFrameDropped=False)

# Prep framestim
disp_size = [x * mag_factor for x in hmd._hmdBufferSize]
framestim = FrameStim(hmd, display_size=disp_size, interpolate=True)

#timestamps = []

# Begin main loop
KEEPGOING = True
while KEEPGOING:
    # Get frame from stream, update and display framestim
    framestim.frame = stream.get_frame()
    framestim.draw()
    t = hmd.flip()
    #timestamps.append(t)

    # Check keys
    for key in event.getKeys():
        if key == 'space':  # switch applying post-proc on space
            stream.switchApplyPostproc()
        elif key in ['escape', 'q']:  # end loop on escape / q
            KEEPGOING = False

# Close
stream.close()
hmd.close()

#import numpy as np
#import matplotlib.pyplot as plt
#d = 1/np.diff(timestamps)
#fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8,4))
#ax1.plot(d)
#ax2.hist(d, 50)
#plt.show()