"""
Main functions and classes
--------------------------
* OpenCV_VideoStream
    Uses generic Open CV commands to provide a set of all purpose methods for
    video capture with any camera.  Easy to use, but may not be able to access
    some camera specific functions (e.g. freerun modes for high frame rates).
* uEyeVideoStream
    Uses pyueye module which provides Python bindings to the uEye SDK.
    Allows running an IDS uEye camera in freerun mode, which will allow acess
    to the higher frame rates.
* FrameStim
    Thin wrapper around PsychoPy ImageStim which allows for providing image as
    a uint8 numpy array and interpolating image to size of PsychoPy window.
* calc_imResize / calc_imCrop
    Functions for working out how to interpolate a camera image up to the
    resolution of a PsychoPy window.

Dependencies
------------
* numpy
* PsychoPy
* OpenCV3 python bindings
* Python Image Library (PIL) or Pillow
* pyueye (available in pip) and the IDS uEye drivers and SDK if wanting to
  use the uEye functions / classes.

"""
from __future__ import division
import os, cv2, warnings
import numpy as np
from psychopy import visual
from psychopy.tools.attributetools import setAttribute
import pyglet.gl as GL

try:
    from pyueye import ueye
    from pyueye.ueye import sizeof
    have_pyueye = True
except ImportError:
    have_pyueye = False

# Py2 <--> Py3 compatibility fixes
from past.builtins import unicode


# Known camera resolutions
cam_res_lookup = {'uEye1':(752,480),
                  'uEye2':(1280,1024),
                  'laptop':(640,480)}



#############################################################################
### Function definitions
#############################################################################
def uEyeCheck(code, action='error', msg=None):
    """
    Handy func for checking ueye return codes for errors.  Action can be one
    of 'error' or 'warn'.  msg can be used to prepend error code with a
    custom message.
    """
    if action not in ['error','warn']:
        raise ValueError("action must be one of 'error' or 'warn'")

    if code != ueye.IS_SUCCESS:
        if msg is not None:
            code = msg + ': ' + str(code)

        if action == 'error':
            raise RuntimeError(code)
        elif action == 'warn':
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn(code)


def calc_imResize(imsize, screensize):
    """
    Calculates size image will need to be to fill the screen whilst maintaining
    the original aspect ratio.  Note this simply returns the required size -
    the actual interpolation of the image to this size needs to be handled
    separately.

    Differs from calc_imCrop in that here the image is resized to fill the
    screen as much as possible but whilst still maintaining the original
    aspect ratio, ensuring none of the image is lost but potentially leaving
    some blank borders at edges of screen.

    Arguments
    ---------
    imsize - (width, height) tuple
        Size of original image.
    screensize - (wdith, height) tuple
        Size of screen to interpolate image to.

    Returns
    -------
    new_imsize (width, height) tuple
        Size image needs to be interpolated to.

    """
    # Ensure numpy array. Also implicitly copies list so we don't accidentally
    # alter original in-place, e.g. psychopy window size
    imsize = np.asarray(imsize)
    screensize = np.asarray(screensize)

    # Calculate W:H ratios
    im_ratio = imsize[0] / imsize[1]
    screen_ratio = screensize[0] / screensize[1]

    # If image and screen already same ratio, just return screen size as is
    if im_ratio == screen_ratio:
        return tuple(screensize.astype(int))
    # If screen more square than image, rescale width to match
    elif screen_ratio < im_ratio:
        return tuple( (imsize * (screensize[0] / imsize[0])).astype(int) )
    # Else screen more rectangular than image, rescale height to match
    else:
        return tuple( (imsize * (screensize[1] / imsize[1])).astype(int) )


def calc_imCrop(imsize, screensize):
    """
    Calculates amount to crop image by so as to match screen aspect ratio.
    Note this simply returns the required cropping slices - the actual
    application of the cropping and the interpolation of the image to the
    screen size needs to be handled separately.

    Differs from calc_imResize in that here we crop the image to match the
    screen aspect ratio, ensuring the image fills the screen, but potentially
    reducing the field of view of the camera.

    Arguments
    ---------
    imsize - (width, height) tuple
        Size of original image.
    screensize - (wdith, height) tuple
        Size of screen to interpolate image to.

    Returns
    -------
    crop_slices - (width, height) tuple of slices
        Pair of slice objects that may be applied to image to crop it to the
        correct aspect ratio.

    """
    # Ensure numpy array. Also implicitly copies list so we don't accidentally
    # alter original in-place, e.g. psychopy window size
    imsize = np.asarray(imsize)
    im_cols, im_rows = imsize
    screensize = np.asarray(screensize)

    # Calculate W:H ratios
    im_ratio = imsize[0] / imsize[1]
    screen_ratio = screensize[0] / screensize[1]

    # Image and screen already same ratio, return slices for full image
    if im_ratio == screen_ratio:
        return (slice(im_rows), slice(im_cols))
    # Else image and screen not same ratio - calculate new size
    else:
        newsize = imsize.copy()
        # If screen more square than image, crop width to match ratio
        if screen_ratio < im_ratio:
            newsize[0] = newsize[1] * screen_ratio
        # Else screen more rectangular than image, crop height to match ratio
        else:
            newsize[1] = newsize[0] / screen_ratio

        # Offsets given by difference between new and old sizes, halved
        offset_cols, offset_rows = ((imsize - newsize) / 2).astype(int)

        # Return slices
        return (slice(offset_rows, im_rows - offset_rows),
                slice(offset_cols, im_cols - offset_cols))




#############################################################################
### Class definitions
#############################################################################

##### Video streaming classes ######
class BaseVideoStream(object):
    """
    Base Class Arguments
    --------------------
    vertical_reverse : bool, optional
        If True, will vertically flip image (default = False).
    horizontal_reverse : bool, optional
        If True, will horizontally flip image (default = False).
    postproc : function, optional
        Function for apply post-processing to frame.  This must accept a numpy
        array with uint8 dtype as its first argument, and return a numpy
        array with uint8 dtype as its only output.  Can be used to apply
        custom manipulations to video stream, such as a delay or a filter.
        Postproc function is only applied as long as the status indicates to;
        this defaults to ON when the class is first initialised and can be
        changed with the .switchApplyPostproc function.  Note that
        post-processing is applied after all other processing steps (including
        image reversals, colour conversions, etc.), but before writing the
        image out to file (if applicable).
    postproc_kwargs : dict, optional
        Dictionary of further keyword arguments to be passed to the postproc
        function.  Should be of form {'arg':val1, 'arg2':val2, etc}.
    warnMissedFrames : bool, optional
        If True (default), a warning will be raised whenever frame acquistion
        fails.

    Base Class Methods
    ------------------
    .get_frame
        Returns a frame from the camera as a numpy array. If image acquisition
        fails, a warning will be printed (if warnMissedFrames == True) and
        None will be returned instead.
    .openVideoWriter
        Open a video writer object to an output file.  Whilst the recording
        status is ON, further calls to .get_frame will also write the frame
        out to the file.  Note - recording status defaults to OFF when stream
        is first initialised.
    .closeVideoWriter
        Close video writer object. Should get called automatically when stream
        is closed.
    .switchRecordingStatus
        Switch the recording status ON or OFF.  Defaults to OFF when class is
        first initialised.
    .switchApplyPostproc
        Switches whether the postproc function (if provided) is applied or not.
        Defaults to ON when class is first initialised.
    .setPostproc
        Supply a new post-processing function.

    Example usage
    -------------
    Examples show usage for OpenCV backend; setup for other backends will
    be similar.

    Create an instance of the video stream.

    >>> from utils.videostreaming import OpenCV_VideoStream
    >>> videostream = OpenCV_VideoStream()

    Calls to the .get_frame method return the frames from the camera. Stick
    these in a loop to continuously acquire.  Here we display them in an
    OpenCV window.

    >>> import cv2
    >>> cv2.namedWindow('display')
    >>> keepgoing = True
    >>> while keepoing:
    ...     frame = videostream.get_frame()
    ...     if frame is not None:
    ...         cv2.imshow('display', frame)
    ...     key = cv2.waitKey(1)
    ...     if key == 27:  # code for escape key
    ...         videostream.close()
    ...         cv2.destroyAllWindows()
    ...         keepgoing = False

    To apply some custom post-processing to the frames, supply a function to
    the class postproc parameter.  Here we create a short function for photo
    negating the frames.  Subsequent calls to the .get_frame method should now
    return the images photo negated.

    >>> def negate(frame):
    ...     return 255 - frame
    >>> videostream = OpenCV_VideoStream(postproc=negate)

    A video writer object can be opened to record the stream to a file.  The
    recording status defaults to OFF when the class is first initialised;
    call .switchRecording to begin recording.

    >>> videostream.openVideoWriter('./test.mp4')
    >>> # When ready, start recording
    >>> videostream.switchRecording()

    Subsequent calls to the .get_frame method will now also write that
    frame out to the video file.  When finished with the recording, the video
    writer must be closed, otherwise the file may not be written out properly.
    If continuing with the stream after recording, the .closeVideoWriter
    function can be used to close the writer specifically.

    >>> videostream.closeVideoWriter()

    Once finished with the stream entirely, the .close method must be called.
    If a video writer is currently open, it will automatically be closed
    (negating the need to call .closeVideoWriter here).

    >>> videostream.close()

    See Also
    --------
    * FrameStim - Thin wrapper around PsychoPy's ImageStim class.  Can be
      used to display frames in a PsychoPy window, with options for cropping
      and / or rescaling the image.

    """

    """
    2nd docstring kept separate from 1st one so that it is not inherited by
    child classes.

    Child classes must provide the following methods:
        * self._acquire_image_data() - Must return a frame as numpy array, or
          return None if image acquisition fails.
        * self.close() - Some method for closing the video stream, which will
          also call this base class's .closeVideoWriter method when it does.

    Child classes must also provide the following attributes
        * self.fps - float giving frames per second
        * self.cam_res - (width, height) tuple of ints of camera resolution
        * self.colour_mode - str giving colour mode, selected from 'bgr',
          'rgb', or 'mono'
    """
    def __init__(self, vertical_reverse=False, horizontal_reverse=False,
                 postproc=None, postproc_kwargs={}, warnMissedFrames=True):
        # Assign local vars into class
        self.vertical_reverse = vertical_reverse
        self.horizontal_reverse = horizontal_reverse
        self.warnMissedFrames = warnMissedFrames

        # Assign postproc func
        self.setPostproc(postproc, postproc_kwargs)

        # Assorted internal flags
        self.RECORDING = False # sets whether to record
        self.APPLYPOSTPROC = True # sets whether to apply postproc func

        # Default video writer to None - will be overwritten with real
        # writer object if one is opened via .openVideoWriter
        self.video_writer = None


    def _acquire_image_data(self):
        """Placeholder function - should be overwritten by child class"""
        raise NotImplementedError


    def close(self):
        """Placeholder function - should be overwritten by child class"""
        raise NotImplementedError


    def get_frame(self):
        """
        Acquire a single frame from the camera and return it the user after
        applying some minimal post-processing plus any additional custom
        post-processing if such a function was specified.

        Returns
        -------
        frame : numpy array with uint8 datatype or None
            If image acquisition is successful, the resulting frame is
            returned as a numpy array. If it fails, a warning will be printed
            (if warnMissedFrames is True) and None will be returned instead.
        """
        # Get frame
        frame = self._acquire_image_data()
        if frame is None:
            if self.warnMissedFrames:
                with warnings.catch_warnings():
                    warnings.simplefilter('always')
                    warnings.warn('Missed camera frame')
            return

        # Flip image upside-down and / or left-right if requested
        if self.vertical_reverse:
            frame = np.flipud(frame)
        if self.horizontal_reverse:
            frame = np.fliplr(frame)

        # If any post-processing requested, apply it now
        if self.postproc and self.APPLYPOSTPROC:
            frame = self.postproc(frame, **self.postproc_kwargs)

        # Write video to output if requested (convert RGB -> BGR if needed)
        if self.video_writer and self.RECORDING:
            if self.colour_mode == 'rgb':
                _frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                _frame = frame
            self.video_writer.write(_frame)

        # Return
        return frame


    def openVideoWriter(self, output_file, codec='mp4v', overwrite=False):
        """
        Opens video writer object for specified output file. Whilst recording
        status is ON, further calls to .get_frame will also write that frame
        out to the file.  Note that the recording status defaults to OFF when
        the class is first initialised, so you will need to switch it ON
        with the .switchRecording function to start writing to the file.

        Arguments
        ---------
        output_file : str, required
            Filepath to desired output video. Recommend using '.mp4' file
            extension (default if no extension provided).
        codec : str or -1, optional
            Fourcc (https://www.fourcc.org/codecs.php) code indicating codec
            to use for encoding video output. Codec must be appropriate for
            file type; recommend using 'mp4v' for MP4 files (default).
        overwrite : bool, optional
            If True, will overwrite output file if it already exists. If
            False (default), will error if output file already exists.

        """
        # Default to .mp4 format
        if not os.path.splitext(output_file)[1]:
            output_file += '.mp4'

        # HACK - opencv seems to mess up attempting to write to an existing
        # file (frames get appended rather than overwritten), so delete
        # file if it already exists
        if os.path.isfile(output_file):
            if overwrite:
                os.remove(output_file)
            else:
                raise IOError('Output video file already exists')

        # Obtain fourcc
        fourcc = cv2.VideoWriter_fourcc(*codec)

        # Create writer object
        self.video_writer = cv2.VideoWriter(
                output_file, fourcc, self.fps, tuple(map(int, self.cam_res)),
                isColor=self.colour_mode != 'mono'
                )

        # Error check
        if not self.video_writer.isOpened():
            raise IOError('Failed to open video writer for {}, check video ' \
                          'file and codec settings'.format(output_file))
        else:
            print('Opened video writer for ' + output_file)


    def closeVideoWriter(self):
        """
        Closes video writer object and hence current output file.  Gets called
        automatically when .close method is called, so should only be
        necessary to use directly if you want to close one video writer and
        open another without stopping the video stream.
        """
        if self.video_writer:
            if self.RECORDING:
                self.switchRecording()
            self.video_writer.release()
            print('Closed video writer')


    def switchRecording(self, value=None):
        """
        Switches recording status ON or OFF. If provided value is a boolean,
        then will set status to this value. If provided value is None
        (default), then will switch to opposite of current status.
        """
        if self.video_writer:
            if value is None:
                self.RECORDING = not self.RECORDING
            else:
                self.RECORDING = value

            if self.RECORDING:
                print('Video stream recording')
            else:
                print('Video stream not recording')
        else:
            pass


    def switchApplyPostproc(self, value=None):
        """
        Switches whether to apply post-processing function ON or OFF.
        If provided value is a boolean, then will set status to this value.
        If provided value is None (default), then will switch to opposite of
        current status.
        """
        if value is None:
            self.APPLYPOSTPROC = not self.APPLYPOSTPROC
        else:
            self.APPLYPOSTPROC = value

        if self.APPLYPOSTPROC:
            print('Applying post-processing')
        else:
            print('Not applying post-processing')


    def setPostproc(self, postproc, postproc_kwargs=None):
        """
        Set post-processing function after class has been initialised, e.g. to
        overwrite old function with a new one mid-stream.  If postproc_kwargs
        is None, existing entry in class will not be changed.
        """
        self.postproc = postproc
        if postproc_kwargs is not None:
            self.postproc_kwargs = postproc_kwargs



class OpenCV_VideoStream(BaseVideoStream):
    """
    Video streaming class based on generic methods implemented in OpenCV. This
    is easy to use, but for professional cameras may not be able to access
    some of the camera's more specific functions.  For example, it will not be
    able to access free-run modes, which may limit the maximum frame rate
    achievable.

    Arguments
    ---------
    cam_num - int, optional
        Index of camera to use.  0 (default) will use first available camera
        (probably laptop camera), 1 will use first available externally
        connected camera.
    cam_res - (width, height) tuple of ints, str, or None, optional
        Resolution to acquire images at.  If tuple of ints, will attempt to
        use that resolution exactly (if the camera supports it).  Can be a
        string giving a named camera listed in the cam_res_lookup table
        included in this module.  If None (default) will attempt to use the
        default resolution retrieved from the camera settings, but note that
        it might not get this right!
    fps - float, optional
        Frames per second to acquire at.  Default of 30 is limit of most
        laptop cameras.
    colour_mode - str {'bgr' | 'rgb' | 'mono'}, optional
        OpenCV acquires images into BGR colour space. Specify colour mode as
        'bgr' to leave the images in this space, as 'rgb' to convert them to
        RGB colour space,  or 'mono' to convert them to grayscale.  Note that
        images are always acquired in BGR and must be converted to other
        spaces, which may incur a small processing cost.
    **kwargs
        Further keyword arguments are passed to the videostreaming base class
        (details of which are included further below).

    Methods
    -------
    .close
        Close video writer (if applicable) and release camera.  Must be called
        when you are done.

    """
    __doc__ += BaseVideoStream.__doc__

    def __init__(self, cam_num=0, cam_res=None, fps=30.0, colour_mode='bgr',
                 **kwargs):
        # Assign local vars into class
        self.cam_num = cam_num
        self.cam_res = cam_res
        self.fps = fps
        self.colour_mode = colour_mode

        # Error check
        if self.colour_mode not in ['bgr','rgb','mono']:
            raise ValueError("colour_mode must be one of 'bgr', 'rgb', "
                             "or 'mono'")

        # If cam_res is named camera, assign known resolution
        if isinstance(self.cam_res, (str, unicode)):
            try:
                self.cam_res = cam_res_lookup[self.cam_res]
            except KeyError:
                raise ValueError('Unrecognised camera: ' + self.cam_res)

        # Acquire video device
        self.cap = cv2.VideoCapture(self.cam_num)
        if not self.cap.isOpened():
            raise IOError('Could not open camera, is it in use by another ' \
                          'process?')

        # Set / get camera resolution
        if self.cam_res:
            # If cam_res provided, set to provided values
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_res[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_res[1])
        else:
            # Otherwise, acquire defaults (tend to be a bit rubbish though!)
            self.cam_res = ( self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                             self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )

        # Set / get camera fps
        if not self.fps:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps == 0:
                self.fps = 30.0
                print('Could not determine camera fps, defaulting to 30')
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Instantiate parent class
        super(OpenCV_VideoStream, self).__init__(**kwargs)


    def _acquire_image_data(self):
        # Acquire frame.  ret will be True if capture was successful
        ret, frame = self.cap.read()
        if ret:
            # Apply colour conversion if necessary
            if self.colour_mode == 'rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif self.colour_mode == 'mono':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Return
            return frame


    def close(self):
        """
        Stop video stream permanently.  You must call this when you are done.
        Function releases the camera capture device, and releases the video
        writer (if applicable).
        """
        self.cap.release()
        self.closeVideoWriter()
        print('Video stream closed')



class uEyeVideoStream(BaseVideoStream):
    """
    Video streaming class based on IDS uEye's SDK.  Requires that pyueye be
    installed on your system (it is available in pip).  Also requires the
    uEye SDK and drivers to be installed.  Camera will be run in freerun mode,
    allowing access to the higher frame rates, but also making operation a bit
    more complicated.  Only compatible with IDS uEye cameras (obviously).

    Arguments
    ---------
    cam_num - int, optional
        Index of uEye camera to use.  Default of 0 selects first available.
    pixel_clock - int, str {'min' | 'max'} or None, optional
        Value to set for the uEye pixel clock, in MHz. This determines the
        bandwidth of the camera data feed.  Larger values allow faster frame
        rates, but excessive values may cause transmission errors.  Ideally
        this should be set just high enough to achieve the desired frame rate.
        The easiest way to determine this is to run the camera via the uEye
        Cockpit application, open the Camera tab under the Camera Properties
        menu, and adjust the pixel clock slider till the maximum achievable
        frame rate is just above the desired value. If 'min' or 'max', will use
        the minimum or maximum values the camera can provide.  If None
        (default), will use the default pixel clock value of the camera; note
        that this will limit the achievable frame rates.  Further details can
        be found within the .pixel_clock_info attribute after initialisation.
    aoi - (left, bottom, width, height) tuple or None, optional
        Area of interest, i.e. region of camera sensor to acquire images
        from.  Could be used to manipulate resolution / aspect ratio.  If None
        (default), will use the maximum allowable AOI.
    fps - float, optional
        Frames per second to acquire (default = 30).  Note that maximum
        achievable frame rate depends on pixel_clock and exposure settings
        (see below).
    colour_mode - str {'bgr' | 'rgb' | 'mono'}, optional
        Colour space to acquire images into.  Will use colour BGR if set as
        'bgr', colour RGB if set as 'rgb', or grayscale if set as 'mono'.
        Note that this determines the space the images are actually acquired
        into - no colour conversion of the images is necessary.
    buffer_size - int or None, optional
        Number of frames to buffer images within.  Larger buffers will incur
        a delay in the stream of images, but buffers which are too small are
        liable to incur memory errors.  If None (default) will use 10% of
        the frame rate (corresponding to 100ms).
    exposure - float, optional
        Camera shutter exposure time in ms.  Default of 0 is a special value
        that sets the exposure to the maximum allowable (1000/fps).  Exposure
        time should not exceed the duration of the frame acquisition at the
        desired fps.  Note that if using auto-exposure then this setting
        will only affect the first few frames acquired.  Further details on
        available options can be found within the .exposure_info attribute
        after initialisation.
    block - bool, optional
        If True, each call to acquire a new frame from the buffer will block
        till the camera has finished acquiring the next one.  If False
        (default), will grab image from buffer without delay - note that if a
        new frame has not yet been acquired it will instead repeatedly return
        the same frame until one has.  Non-blocking operation is recommended
        if running a live video feed on screen through PsychoPy at higher
        camera frame rates (e.g. camera FPS matches the monitor rate) because
        PsychoPy's win.flip() command is itself usually a blocking operation.
        If not streaming the video via PsychoPy (e.g. video is simply being
        recorded to file, or just a simple stream is being displayed with
        OpenCV), or if the camera frame rate is less than the monitor rate,
        then blocking operation is likely to be preferable.
    auto_exposure - str {'camera' | 'software'} or False, optional
        If not False, will automatically adjust shutter exposure to try and
        maintain a middling luminance level for the frames.  Default is False.
        See notes below for meaning of string options.
    auto_gain_control - str {'camera' | 'software'} or False, optional
        If not False, will automatically adjust the camera's gain control to
        try and maintain a middling luminance level for the frames. Default is
        False.  See notes below for meaning of string options.
    auto_white_balance - str {'camera' | 'software'} or False, optional
        If not False, will compare luminance levels across colour channels to
        try and maintain a middling luminance level for the frames.  Only
        available for colour cameras, and only if images are being acquired in
        a colour space.  Default is False.  See notes below for meaning of
        string options.
    colour_correction - bool, optional
        If True, will apply a correction that enhances colours to produce
        a more vibrant looking colour display.  Would recommend using in
        conjunction with auto white balance. Only available on certain cameras,
        and only if images are being acquired in a colour space. Default is
        False.  See also colour_correction_factor.
    colour_correction_factor - float, optional
        Value between 0 (min) and 1 (max) indicating strength of colour
        correction to apply (default = 0.5).  Ignored if colour_correction
        is False.
    **kwargs
        Further keyword arguments are passed to the videostreaming base class
        (details of which are included further below).

    Auto-control options
    --------------------
    Auto-control options accept the special strings 'software' or 'camera'.
    These determine the point at which the auto-control is performed.  If
    'software', the control is performed by the computer.  If 'camera', the
    control is performed by the camera itself.  Camera control is generally
    preferred as it reduces the processing cost to the computer, but note that
    some camera models may not support this type of control for some options.

    Modifying further options
    -------------------------
    The uEye SDK allows for modifying many more options than are listed in
    this class.  It is possible to further modify other options after the
    class has been initialised using the pyueye module to access the relevant
    SDK functions.  Most functions require the camera handle to be passed as a
    parameter - this can be obtained from the .cam attribute of this class
    after  initialisation.  Details of the SDK commands available can be found
    in the uEye manual: https://en.ids-imaging.com/manuals-ueye-software.html

    Methods
    -------
    .update_pixel_clock_info
        Update internal store of pixel clock info.
    .update_exposure_info
        Update internal store of exposure info.
    .start_freerun
        Begin freerun mode; will get called automatically whenever the first
        frame is requested.
    .stop_freerun
        Stop freerun mode; will get called automatically when the stream is
        closed, but will also need to be called manually if the stream needs
        to be paused at any point.
    .close
        Close video writer (if applicable) and release camera.  Must be called
        when you are done.

    """
    __doc__ += BaseVideoStream.__doc__

    def __init__(self, cam_num=0, pixel_clock=None, fps=30.0, aoi=None,
                 colour_mode='bgr',  buffer_size=None,  exposure=0.0,
                 block=False, auto_exposure=False, auto_gain_control=False,
                 auto_white_balance=False, colour_correction=False,
                 colour_correction_factor=0.5, **kwargs):

        # Make sure we have pyueye module
        if not have_pyueye:
            raise RuntimeError('Missing pyueye module')

        # Assign args into class.  Some of these need to be cast to ctypes
        # objects but we will do this later so as to allow allocating other
        # default values first.
        self.cam_num = cam_num
        self.pixel_clock = pixel_clock
        self.fps = fps
        self.aoi = aoi
        self.colour_mode = colour_mode.lower()
        self.buffer_size = buffer_size
        self.exposure = exposure
        self.block = block
        self.auto_exposure = auto_exposure
        self.auto_gain_control = auto_gain_control
        self.auto_white_balance = auto_white_balance
        self.colour_correction = colour_correction
        self.colour_correction_factor = colour_correction_factor

        # Other internal variables
        self.RUNNING = False

        # Error check
        if self.colour_mode not in ['bgr','rgb','mono']:
            raise ValueError("colour_mode must be one of 'bgr', 'rgb', "
                             "or 'mono'")
        if self.auto_exposure not in ['camera', 'software', False]:
            raise ValueError("auto_exposure must be one of 'camera', "
                             "'software', or False")
        if self.auto_gain_control not in ['camera', 'software', False]:
            raise ValueError("auto_gain_control must be one of 'camera', "
                             "'software', or False")
        if self.auto_white_balance not in ['camera', 'software', False]:
            raise ValueError("auto_white_balance must be one of 'camera', "
                             "'software', or False")

        # Set buffer_size as 10% of fps if not specified
        if self.buffer_size is None:
            self.buffer_size = max(1, int(round(0.1 * self.fps)))

        # Open handle to specified camera and initialise
        self.cam = ueye.HIDS(self.cam_num)
        uEyeCheck(ueye.is_InitCamera(self.cam, None), msg='InitCamera')

        # Get sensor info
        self.sensorInfo = ueye.SENSORINFO()
        uEyeCheck(ueye.is_GetSensorInfo(self.cam, self.sensorInfo),
                  msg='GetSensorInfo')

        # Ensure set to freerun mode
        uEyeCheck(ueye.is_SetExternalTrigger(self.cam, ueye.IS_SET_TRIGGER_OFF),
                  msg='SetExternalTrigger')

        # Set area of interest.  If not provided, use max allowable values
        # recovered from sensor info
        self.aoi_rect = ueye.IS_RECT()
        if self.aoi is not None:
            self.aoi_rect.s32X = ueye.int(self.aoi[0])
            self.aoi_rect.s32Y = ueye.int(self.aoi[1])
            self.aoi_rect.s32Width = ueye.int(self.aoi[2])
            self.aoi_rect.s32Height = ueye.int(self.aoi[3])
        else:
            self.aoi_rect.s32X = ueye.int(0)
            self.aoi_rect.s32Y = ueye.int(0)
            self.aoi_rect.s32Width = ueye.int(self.sensorInfo.nMaxWidth)
            self.aoi_rect.s32Height = ueye.int(self.sensorInfo.nMaxHeight)
        uEyeCheck(ueye.is_AOI(self.cam, ueye.IS_AOI_IMAGE_SET_AOI,
                               self.aoi_rect, sizeof(self.aoi_rect)),
                  msg='Set AOI')

        # Work out camera resolution from AOI
        self.cam_res = (self.aoi_rect.s32Width.value,
                        self.aoi_rect.s32Height.value)

        # Set colour mode and corresponding bits per pixel
        if self.colour_mode == 'bgr':
            uEyeCheck(ueye.is_SetColorMode(self.cam, ueye.IS_CM_BGR8_PACKED),
                      msg='SetColorMode')
            self.bits_per_pixel = 24
        elif self.colour_mode == 'rgb':
            uEyeCheck(ueye.is_SetColorMode(self.cam, ueye.IS_CM_RGB8_PACKED),
                      msg='SetColorMode')
            self.bits_per_pixel = 24
        elif self.colour_mode == 'mono':
            uEyeCheck(ueye.is_SetColorMode(self.cam, ueye.IS_CM_MONO8),
                      msg='SetColorMode')
            self.bits_per_pixel = 8

        # Cast pixel clock to ctypes, set in ueye
        self.update_pixel_clock_info()  # to get min / max vals
        if self.pixel_clock is not None:
            if self.pixel_clock == 'min':
                self.c_pixel_clock = self.pixel_clock_info.clockMin
            elif self.pixel_clock == 'max':
                self.c_pixel_clock = self.pixel_clock_info.clockMax
            else:
                self.c_pixel_clock = ueye.uint(self.pixel_clock)
            uEyeCheck(ueye.is_PixelClock(
                    self.cam, ueye.IS_PIXELCLOCK_CMD_SET,
                    self.c_pixel_clock, sizeof(self.c_pixel_clock)
                    ), msg='Set PixelClock')

        # Cast fps to ctypes, set in ueye
        self.c_fps = ueye.double(self.fps)
        actual_fps = ueye.double()
        uEyeCheck(ueye.is_SetFrameRate(self.cam, self.c_fps, actual_fps),
                  msg='Set FrameRate')
        if not abs(self.c_fps - actual_fps) < 1.0:
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn('Could not achieve desired fps - largest '
                              'available was {}. Try adjusting pixel clock.'
                              .format(actual_fps))

        # Cast exposure to ctypes, set in ueye
        self.c_exposure = ueye.double(self.exposure)
        uEyeCheck(ueye.is_Exposure(
                self.cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                self.c_exposure, sizeof(self.c_exposure)
                ), msg='Set Exposure')

        # Set auto control settings
        if self.auto_exposure == 'camera':
            uEyeCheck(ueye.is_SetAutoParameter(
                    self.cam, ueye.IS_SET_ENABLE_AUTO_SENSOR_SHUTTER,
                    ueye.double(True), ueye.double()
                    ), msg='Set auto exposure')
        elif self.auto_exposure == 'software':
                 uEyeCheck(ueye.is_SetAutoParameter(
                    self.cam, ueye.IS_SET_ENABLE_AUTO_SHUTTER,
                    ueye.double(True), ueye.double()
                    ), msg='Set auto exposure')

        if self.auto_gain_control == 'camera':
            uEyeCheck(ueye.is_SetAutoParameter(
                    self.cam, ueye.IS_SET_ENABLE_AUTO_SENSOR_GAIN,
                    ueye.double(True), ueye.double()
                    ), msg='Set auto gain control')
        elif self.auto_gain_control == 'software':
            uEyeCheck(ueye.is_SetAutoParameter(
                    self.cam, ueye.IS_SET_ENABLE_AUTO_GAIN,
                    ueye.double(True), ueye.double()
                    ), msg='Set auto gain control')

        if self.colour_mode != 'mono':
            if self.auto_white_balance == 'camera':
                uEyeCheck(ueye.is_SetAutoParameter(
                        self.cam, ueye.IS_SET_ENABLE_AUTO_SENSOR_WHITEBALANCE,
                        ueye.double(ueye.WB_MODE_AUTO), ueye.double()
                        ), msg='Set auto white-balance')
            elif self.auto_white_balance == 'software':
                uEyeCheck(ueye.is_SetAutoParameter(
                        self.cam, ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE,
                        ueye.double(True), ueye.double()
                        ), msg='Set auto white-balance')

            if self.colour_correction:
                uEyeCheck(ueye.is_SetColorCorrection(
                        self.cam, ueye.IS_CCOR_ENABLE_NORMAL,
                        ueye.double(self.colour_correction_factor)
                        ), msg='Set colour correction')

        # Allocate image buffers.  Note that the list of buffers is only used
        # at end when we clear them - ueye allocates data to the buffers
        # internally, so we don't interact with them directly otherwise
        self.image_buffers = []
        for i in range(self.buffer_size):
            buff = uEyeImageBuffer()
            uEyeCheck(ueye.is_AllocImageMem(
                    self.cam, self.aoi_rect.s32Width, self.aoi_rect.s32Height,
                    self.bits_per_pixel, buff.mem_ptr, buff.mem_id
                    ), msg='AllocImageMem')
            uEyeCheck(ueye.is_AddToSequence(self.cam, buff.mem_ptr, buff.mem_id),
                      msg='AddToSequence')
            self.image_buffers.append(buff)
        uEyeCheck(ueye.is_InitImageQueue(self.cam, 0), msg='InitImageQueue')

        # Create one more image buffer which will be used to receive the
        # incoming images during video acquistion
        self.receptor_buffer = uEyeImageBuffer()

        # Query pitch - gives num image rows * num colour channels
        self.c_pitch = ueye.int()
        uEyeCheck(ueye.is_GetImageMemPitch(self.cam, self.c_pitch),
                  msg='GetImageMemPitch')

        # Update info now all settings have been set
        self.update_pixel_clock_info()
        self.update_exposure_info()

        # Instantiate base class
        super(uEyeVideoStream, self).__init__(**kwargs)


    def _acquire_image_data(self):
        # Begin image capture if necessary
        if not self.RUNNING:
            self.start_freerun()

        # Get pointer to currently active image buffer, dependent on blocking
        if self.block:
            ueye.is_WaitForNextImage(
                    self.cam, 1000, self.receptor_buffer.mem_ptr,
                    self.receptor_buffer.mem_id
                    )
        else:
            ueye.is_GetImageMem(self.cam, self.receptor_buffer.mem_ptr)

        # Collect image data into numpy array (might fail, e.g. if memory
        # resource not yet free)
        try:
            frame = ueye.get_data(
                    self.receptor_buffer.mem_ptr, self.aoi_rect.s32Width,
                    self.aoi_rect.s32Height, self.bits_per_pixel,
                    self.c_pitch, copy=True
                    )
            if self.colour_mode in ['bgr','rgb']:
                frame = frame.reshape(self.aoi_rect.s32Height.value,
                                      self.aoi_rect.s32Width.value,
                                      3)
            elif self.colour_mode == 'mono':
                frame = frame.reshape(self.aoi_rect.s32Height.value,
                                      self.aoi_rect.s32Width.value)
        except Exception as e:
            print(e)
            frame = None

        # Need to unlock buffer resource so it can be used again
        ueye.is_UnlockSeqBuf(self.cam, self.receptor_buffer.mem_id,
                             self.receptor_buffer.mem_ptr)

        # Return
        return frame


    def update_pixel_clock_info(self):
        self.pixel_clock_info = uEyePixelClockInfo(self.cam)


    def update_exposure_info(self):
        self.exposure_info = uEyeExposureInfo(self.cam)


    def start_freerun(self):
        """
        Start freerun mode. Should get called automatically whenever the first
        frame is requested.
        """
        uEyeCheck(ueye.is_CaptureVideo(self.cam, ueye.IS_WAIT),
                  msg='CaptureVideo')
        self.RUNNING = True
        print('Video freerun started')


    def stop_freerun(self):
        """
        Stop / pause freerun mode.  Should get called automatically when the
        stream is closed, but will also need to called if the stream needs to
        be paused at any point.
        """
        uEyeCheck(ueye.is_StopLiveVideo(self.cam, ueye.IS_FORCE_VIDEO_STOP),
                  action='warn', msg='StopLiveVideo')
        self.RUNNING = False
        print('Video freerun stopped')


    def close(self):
        """Close video stream permanently"""
        # We need as much of this code to execute as possible to try and
        # ensure we release as many resources as we can, so we opt to warn
        # about error codes rather than raise full errors

        # Stop video stream
        if self.RUNNING:
            self.stop_freerun()

        # Clear memories from sequence (created using AddToSequence)
        uEyeCheck(ueye.is_ClearSequence(self.cam), action='warn',
                  msg='ClearSequence')

        # Release memory resources for image buffers
        for buff in self.image_buffers:
            uEyeCheck(ueye.is_FreeImageMem(self.cam, buff.mem_ptr, buff.mem_id),
                      action='warn', msg='FreeImageMem')

        # Delete image queue (initialised with InitImageQueue)
        uEyeCheck(ueye.is_ExitImageQueue(self.cam), action='warn',
                  msg='ExitImageQueue')

        # Release camera handle
        uEyeCheck(ueye.is_ExitCamera(self.cam), action='warn',
                  msg='ExitCamera')

        # Close video writer if applicable
        self.closeVideoWriter()

        # Done
        print('Closed video stream')




##### Other (not directly videostreaming) classes ######
class uEyeImageBuffer(object):
    """
    Simple container providing a memory pointer and id, to use as an image
    buffer.
    """
    def __init__(self):
        self.mem_ptr = ueye.c_mem_p()
        self.mem_id = ueye.int()



class uEyePixelClockInfo(object):
    """Containter for all things fun about the uEye pixel clock."""
    def __init__(self, cam):
        # Number of available clocks
        self.nAvailableClocks = ueye.uint()
        uEyeCheck(ueye.is_PixelClock(
                cam, ueye.IS_PIXELCLOCK_CMD_GET_NUMBER,
                self.nAvailableClocks, sizeof(self.nAvailableClocks)
                ), msg='Get num PixelClocks')

        # Possible range of values for clock.  If increment is 0, indicates
        # that only discrete values are allowed, in which case we also create
        # a list of the available clocks.
        self.clockRange = (ueye.uint * 3)(0,0,0)
        uEyeCheck(ueye.is_PixelClock(
                cam, ueye.IS_PIXELCLOCK_CMD_GET_RANGE,
                self.clockRange, sizeof(self.clockRange)
                ), msg='Get PixelClock range')
        self.clockMin, self.clockMax, self.clockIncrement = self.clockRange

        if self.clockIncrement == 0:  # discrete clocks
            self.clockList = (ueye.uint * self.nAvailableClocks.value) \
                             (*([0]*self.nAvailableClocks.value))
            uEyeCheck(ueye.is_PixelClock(
                    cam, ueye.IS_PIXELCLOCK_CMD_GET_LIST,
                    self.clockList, sizeof(self.clockList)
                    ), msg='Get PixelClock list')
        else:
            self.clockList = None

        # Query default clock
        self.defaultClock = ueye.uint()
        uEyeCheck(ueye.is_PixelClock(
                cam, ueye.IS_PIXELCLOCK_CMD_GET_DEFAULT,
                self.defaultClock, sizeof(self.defaultClock)
                ), msg='Get default PixelClock')

        # Query current clock
        self.currentClock = ueye.uint()
        uEyeCheck(ueye.is_PixelClock(
                cam, ueye.IS_PIXELCLOCK_CMD_GET,
                self.currentClock, sizeof(self.currentClock)
                ), msg='Get current PixelClock')



class uEyeExposureInfo(object):
    """Container for all things fun about the uEye exposure settings."""
    def __init__(self, cam):
        # Query min, max, and increment
        self.exposureRange = (ueye.double * 3) (0,0,0)
        uEyeCheck(ueye.is_Exposure(
                cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE,
                self.exposureRange, sizeof(self.exposureRange)
                ), msg='Get exposure range')
        self.exposureMin, self.exposureMax, self.exposureIncrement = \
            self.exposureRange

        # Query default
        self.defaultExposure = ueye.double()
        uEyeCheck(ueye.is_Exposure(
                cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_DEFAULT,
                self.defaultExposure, sizeof(self.defaultExposure)
                ), msg='Get default exposure')

        # Query current exposure
        self.currentExposure = ueye.double()
        uEyeCheck(ueye.is_Exposure(
                cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
                self.currentExposure, sizeof(self.currentExposure)
                ), msg='Get current exposure')


class FrameStim(visual.ImageStim):
    """
    Wrapper around psychopy.visual.ImageStim class.  Allows specifying input
    images as numpy arrays in range 0:255 with uint8 datatype (normal
    ImageStim requires them to be floats in range -1:1 or 0:1), and provides
    options for cropping and / or rescaling the images to fit a given display
    size. Note that in doing this, some of the added functionality of the
    normal ImageStim (e.g. ability to apply masks, etc.) is broken.

    Arguments
    ---------
    win - psychopy window instance, required
        Window that frames should be attached to.
    frame - numpy array with uint8 datatype or None, optional
        Image to display.
    rescale - str {'resize' | 'crop'}, None, or False, optional
        Method to use for rescaling.  If 'resize' (default) image is rescaled
        to fill as much of the specified display size as possible, but whilst
        maintaining the original image aspect ratio.  If 'crop', image is
        cropped then rescaled to fill the specfied display size completely.
        If None or False, no rescaling is performed and the image is drawn at
        its original resolution.
    display_size = None, float, or (W,H) tuple of ints, optional
        Target size to display image at, allowing for specified rescaling
        type.  If None (default), uses the PsychoPy window size.  If a float,
        uses a proportion of the PsychoPy window size as given by this value.
        If a (width, height) tuple of ints, uses that size directly (assumes
        values are in pixel units).  Has no effect if rescale is None.
    **kwargs
        Further keyword arguments are passed to the PsychoPy ImageStim base
        class.

    Methods
    -------
    .setFrame
        Update frame image.  Can also assign directly by calling
        myFrameStim.frame = <whatever>.

    """
    def __init__(self, win, frame=None, rescale='resize', display_size=None,
                 **kwargs):
        # Assign local vars into class
        self.win = win
        self.rescale = rescale

        # Error check
        if self.rescale and self.rescale not in ['resize', 'crop']:
            raise ValueError('Invalid argument to \'rescale\'')

        # Assign display size
        if display_size is None:
            self.display_size = self.win.size
        elif isinstance(display_size, (int, float)):
            self.display_size = tuple(x * display_size for x in self.win.size)
        elif hasattr(display_size, '__iter__') and len(display_size) == 2:
            self.display_size = display_size
        else:
            raise ValueError('Invalid value to \'display_size\'')

        # Other internal variables
        self._frame_size = None  # needs to be separate from ImageStim.size
        self._frame_data = None

        # Set / overwrite kwargs we need set to particular values
        kwargs['units'] = 'pix'
        kwargs['size'] = None
        kwargs['flipVert'] = True

        # Instantiate parent class
        super(FrameStim, self).__init__(self.win, **kwargs)

        # Assign frame - need to do this last so that setter method can
        # access all the attributes it needs
        self.frame = frame


    def _createTexture(self, data, id, pixFormat, stim, res=None,
                       maskParams=None, forcePOW2=True, dataType=None):
        """
        Monkey-patch over psychopy.visual.basevisual.TextureMixin function.
        Allows passing uint8 numpy array in range 0:255, and removes many
        of the extra bells and whistles normally in ImageStim so that we can
        render faster.

        :params:
            data:
                numpy array
            id:
                is the texture ID
            stim:
                ImageStim instance
            pixFormat, res, maskParams, forcePOW2, dataType:
                maintained for compatibility with ImageStim, but no longer do
                anything
        """
        if data is None:
            return

        if data.dtype != np.ubyte or (data.min() < 0 or data.max() > 255):
            raise TypeError('Input array should be uint8 in range 0:255')
        useShaders = stim.useShaders
        interpolate = stim.interpolate
        dataType = GL.GL_UNSIGNED_BYTE
        stim._tex1D = False

        # Hack - psychopy expects RGB images to be unsigned, but mono images
        # to be signed.  We can trick psychopy into doing the right thing
        # by just telling it the image is RGB (wasLum=False) regardless of
        # whether or not it actually is.
        wasLum = False

        # handle a numpy array
        if len(data.shape) == 3:  # RGB(A)
            if data.shape[2] == 4:  # RGBA
                pixFormat = internalFormat = GL.GL_RGBA
            else:  # hopefully RGB
                pixFormat = internalFormat = GL.GL_RGB
        else:  # mono
            pixFormat = internalFormat = GL.GL_LUMINANCE

        # serialise
        texture = data.ctypes

        # bind the texture in openGL
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, id)  # bind that name to the target
        # makes the texture map wrap (this is actually default anyway)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        # data from PIL/numpy is packed, but default for GL is 4 bytes
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        # important if using bits++ because GL_LINEAR
        # sometimes extrapolates to pixel vals outside range
        if interpolate:
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            if useShaders:
                # GL_GENERATE_MIPMAP was only available from OpenGL 1.4
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_GENERATE_MIPMAP,
                                   GL.GL_TRUE)
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internalFormat,
                                data.shape[1], data.shape[0], 0,
                                pixFormat, dataType, texture)
            else:  # use glu
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                                   GL.GL_LINEAR_MIPMAP_NEAREST)
                GL.gluBuild2DMipmaps(GL.GL_TEXTURE_2D, internalFormat,
                                     data.shape[1], data.shape[0],
                                     pixFormat, dataType, texture)
        else:
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internalFormat,
                            data.shape[1], data.shape[0], 0,
                            pixFormat, dataType, texture)
        GL.glTexEnvi(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE,
                     GL.GL_MODULATE)  # ?? do we need this - think not!
        # unbind our texture so that it doesn't affect other rendering
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return wasLum


    @property
    def frame(self):
        return self.image


    @frame.setter
    def frame(self, img):
        """
        setter decorator allows us to modify behaviour of any calls to
        obj.frame = <whatever>.  Specifically, we provide options for
        interpolating the image to a different res before passing it to the
        underlying ImageStim.
        """
        # If img is None, return immediately
        if img is None:
            return

        # Calculate size to interpolate to if necessary
        if self._frame_size is None:
            if not self.rescale:
                self._frame_size = self.size = img.shape[:2][::-1]
            elif self.rescale == 'resize':
                self._frame_size = self.size = calc_imResize(
                        img.shape[:2][::-1], self.display_size
                        )
            elif self.rescale == 'crop':
                self.crop_slices = calc_imCrop(
                        img.shape[:2][::-1], self.display_size
                        )
                self._frame_size = self.size = self.display_size

        # Crop image if necessary
        if self.rescale == 'crop':
            img = img[self.crop_slices]

        # Pass to underlying imagestim object. Need to pass a copy otherwise
        # we risk access violations, reshaping errors, etc. Data is copied to
        # an intermediate array first then set to imagestim to ensure image
        # setter method is called properly. On subsequent calls we can save a
        # bit of time by just copying into existing array.
        if self._frame_data is None:
            self._frame_data = img.copy()
        else:
            np.copyto(self._frame_data, img)
        self.image = self._frame_data


    def setFrame(self, img, log=None):
        setAttribute(self, 'frame', img, log)

