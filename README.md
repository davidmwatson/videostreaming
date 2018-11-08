# videostreaming
Python tools for streaming video from a camera and displaying feed on an Oculus Rift, accomplished via [PsychoPy](http://www.psychopy.org) [1,2] and Matthew Cutone's [psychxr module](https://github.com/mdcutone/psychxr) [3] + accompanying PsychoPy tools.

## Scripts
* videostreaming.py - Provides tools for acquiring images from camera, and for formatting these images for display in PsychoPy.
* runDemo.py - A demonstration of using the above tools to run a video feed and display the images on an Oculus Rift. Also shows option of applying further post-processing to manipulate images (in this case, photo negating them). 

## Camera backends
Two backends are provided for acquiring images from the camera:
* OpenCV - Fairly general purpose, should work with most cameras and manufacturers. Might not be able to access some manufacturer-specific camera features (e.g. free-run modes).
* uEye SDK - Will only work for IDS uEye cameras, but does allow access to more specialised functions of those cameras.

It should be relatively easy to write other backends by sub-classing the base video streaming class.

## Dependencies
* Windows OS
* Python3, 64-bit (code has been tested for Python 3.6)
* [PsychoPy](http://www.psychopy.org/) (version >= 3.0). Note that the PsychoPy standalone cannot currently be used as this isn't available for 64-bit python.
* [psychxr module](https://github.com/mdcutone/psychxr)
* Python bindings to OpenCV3
* [pyueye module](https://pypi.org/project/pyueye/) along with uEyE drivers and SDK if wanting to use the uEye camera backend.
* [Oculus Rift Software](https://www.oculus.com/setup/) - note that this will need to be open at all times to use the Oculus Python tools.

## References
[1] Peirce, JW (2007) PsychoPy - Psychophysics software in Python. J Neurosci Methods, 162(1-2):8-13

[2] Peirce JW (2009) Generating stimuli for neuroscience using PsychoPy. Front. Neuroinform. 2:10. doi:10.3389/neuro.11.010.2008

[3] Cutone, M. D. & Wilcox, L. M. (2018). PsychXR (Version 0.1.4) [Software]. Available from https://github.com/mdcutone/psychxr
