#include "libuvc/libuvc.h"
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <unistd.h>
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging




// See all the available parameter options withe the `--help` flag. E.g. `./build/examples/openpose/openpose.bin --help`.
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path,               "examples/media/COCO_val2014_000000000192.jpg",     "Process the desired image.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased,"
                                                        " the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect"
                                                        " ratio possible to the images or videos to be processed. E.g. the default `656x368` is"
                                                        " optimal for 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) videos.");
DEFINE_string(resolution,               "1280x720",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " default images resolution.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                                        " will only display the results on a black background.");
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");

op::CvMatToOpInput *cvMatToOpInput;
op::CvMatToOpOutput *cvMatToOpOutput;
op::PoseExtractorCaffe *poseExtractorCaffeL;
op::PoseRenderer *poseRendererL;
op::OpOutputToCvMat *opOutputToCvMatL;
op::OpOutputToCvMat *opOutputToCvMatR;

bool keep_on = true;
bool inited = false;


void splitVertically(const cv::Mat & input, cv::Mat & outputleft, cv::Mat & outputright)
{

  int rowoffset = input.rows;
  int coloffset = input.cols / 2;

  int r = 0;
  int c = 0;

  outputleft = input(cv::Range(r, std::min(r + rowoffset, input.rows)), cv::Range(c, std::min(c + coloffset, input.cols)));

  c += coloffset;

  outputright = input(cv::Range(r, std::min(r + rowoffset, input.rows)), cv::Range(c, std::min(c + coloffset, input.cols)));
    
}


/* This callback function runs once per frame. Use it to perform any
 * quick processing you need, or have it put the frame into your application's
 * input queue. If this function takes too long, you'll start losing frames. */
void cb(uvc_frame_t *frame, void *ptr) {
  uvc_frame_t *bgr;
  uvc_error_t ret;

  /* We'll convert the image from YUV/JPEG to BGR, so allocate space */
  bgr = uvc_allocate_frame(frame->width * frame->height * 3);
  if (!bgr) {
    printf("unable to allocate bgr frame!");
    return;
  }

  /* Do the BGR conversion */
  ret = uvc_any2bgr(frame, bgr);
  if (ret) {
    uvc_perror(ret, "uvc_any2bgr");
    uvc_free_frame(bgr);
    return;
  }

  // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
  if (inited == false)
  {
    poseExtractorCaffeL->initializationOnThread();
    poseRendererL->initializationOnThread();
    inited = true;
  }
   
  IplImage* cvImg = cvCreateImageHeader(
        cvSize(bgr->width, bgr->height),
       IPL_DEPTH_8U,
       3);
     
  cvSetData(cvImg, bgr->data, bgr->width * 3); 

  cv::Mat image = cv::cvarrToMat(cvImg);
  cv::Mat imageleft;
  cv::Mat imageright;

  //Split image  vertically into 2 even parts
  splitVertically(image, imageleft, imageright);

  
  // Step 2 - Format input image to OpenPose input and output formats
  op::Array<float> netInputArrayL;
  op::Array<float> netInputArrayR;
  op::Array<float> outputArrayL;
  op::Array<float> outputArrayR;
  std::vector<float> scaleRatiosL;
  std::vector<float> scaleRatiosR;
  double scaleInputToOutputL;
  double scaleInputToOutputR;

  std::tie(netInputArrayL, scaleRatiosL) = cvMatToOpInput->format(imageleft);
  std::tie(scaleInputToOutputL, outputArrayL) = cvMatToOpOutput->format(imageleft);
  std::tie(netInputArrayR, scaleRatiosR) = cvMatToOpInput->format(imageright);
  std::tie(scaleInputToOutputR, outputArrayR) = cvMatToOpOutput->format(imageright);

  // Step 3 - Estimate poseKeypoints
  poseExtractorCaffeL->forwardPass(netInputArrayL, {imageleft.cols, imageleft.rows}, scaleRatiosL);
  const auto poseKeypointsL = poseExtractorCaffeL->getPoseKeypoints();
  poseExtractorCaffeL->forwardPass(netInputArrayR, {imageright.cols, imageright.rows}, scaleRatiosR);
  const auto poseKeypointsR = poseExtractorCaffeL->getPoseKeypoints();

  // Step 4 - Render poseKeypoints
  poseRendererL->renderPose(outputArrayL, poseKeypointsL);
  poseRendererL->renderPose(outputArrayR, poseKeypointsR);    
  
  // Step 5 - OpenPose output format to cv::Mat
  auto outputImageL = opOutputToCvMatL->formatToCvMat(outputArrayL);
  auto outputImageR = opOutputToCvMatL->formatToCvMat(outputArrayR);

  // ------------------------- SHOWING RESULT -------------------------
  
  cv::namedWindow("Test L", CV_WINDOW_AUTOSIZE);
  cv::imshow("Test L", outputImageL);

  cv::namedWindow("Test R", CV_WINDOW_AUTOSIZE);
  cv::imshow("Test R", outputImageR);

  int k = cvWaitKey(10);
  if (k == 27)
  {
      keep_on = false;
  }
   
  cvReleaseImageHeader(&cvImg);
   
  uvc_free_frame(bgr);
}

int main(int argc, char **argv) {
  uvc_context_t *ctx;
  uvc_device_t *dev;
  uvc_device_handle_t *devh;
  uvc_stream_ctrl_t ctrl;
  uvc_error_t res;

  /* Initialize a UVC service context. Libuvc will set up its own libusb
   * context. Replace NULL with a libusb_context pointer to run libuvc
   * from an existing libusb context. */
  res = uvc_init(&ctx, NULL);

  if (res < 0) {
    uvc_perror(res, "uvc_init");
    return res;
  }

  puts("UVC initialized");

  /* Locates the first attached UVC device, stores in dev */
  res = uvc_find_device(
      ctx, &dev,
      0, 0, NULL); /* filter devices: vendor_id, product_id, "serial_num" */

  if (res < 0) {
    uvc_perror(res, "uvc_find_device"); /* no devices found */
  } else {
    puts("Device found");

    /* Try to open the device: requires exclusive access */
    res = uvc_open(dev, &devh);

    if (res < 0) {
      uvc_perror(res, "uvc_open"); /* unable to open device */
    } else {
      puts("Device opened");

      /* Print out a message containing all the information that libuvc
       * knows about the device */
      //uvc_print_diag(devh, stderr);

      /* Try to negotiate a 640x480 30 fps YUYV stream profile */
      res = uvc_get_stream_ctrl_format_size(
          devh, &ctrl, /* result stored in ctrl */
          UVC_FRAME_FORMAT_YUYV, /* YUV 422, aka YUV 4:2:2. try _COMPRESSED */
          //1344, 376, 100
          2560, 720, 60 /* width, height, fps */
      );

      // Step 2 - Read Google flags (user defined configuration)
      // outputSize
      const auto outputSize = op::flagsToPoint(FLAGS_resolution, "2560x720");
      // netInputSize
      const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "2560x720");
      // netOutputSize
      const auto netOutputSize = netInputSize;
      // poseModel
      const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);

      // Step 3 - Initialize all required classes
      cvMatToOpInput = new op::CvMatToOpInput{netInputSize, FLAGS_scale_number, (float)FLAGS_scale_gap};
      cvMatToOpOutput = new op::CvMatToOpOutput{outputSize};
      poseExtractorCaffeL = new op::PoseExtractorCaffe{netInputSize, netOutputSize, outputSize, FLAGS_scale_number, poseModel,
                                                    FLAGS_model_folder, FLAGS_num_gpu_start};
      poseRendererL = new op::PoseRenderer{netOutputSize, outputSize, poseModel, nullptr, (float)FLAGS_render_threshold,
                                        !FLAGS_disable_blending, (float)FLAGS_alpha_pose};
      opOutputToCvMatL = new op::OpOutputToCvMat{outputSize};
      opOutputToCvMatR = new op::OpOutputToCvMat{outputSize};


      /* Print out the result */
      uvc_print_stream_ctrl(&ctrl, stderr);

      if (res < 0) {
        uvc_perror(res, "get_mode"); /* device doesn't provide a matching stream */
      } else {  
        /* Start the video stream. The library will call user function cb:
         *   cb(frame, (void*) 12345)
         */
        res = uvc_start_streaming(devh, &ctrl, cb, (void*)123450, 0);

        if (res < 0) {
          uvc_perror(res, "start_streaming"); /* unable to start stream */
        } else {
          puts("Streaming...");

          uvc_set_ae_mode(devh, 1); /* e.g., turn on auto exposure */

          /*wait for an environment variable to be set */
          //sleep(10); /* stream for 10 seconds */
          while(keep_on)
          {
            sleep(1);
          }

          /* End the stream. Blocks until last callback is serviced */
          uvc_stop_streaming(devh);
          puts("Done streaming.");
          cv::destroyAllWindows();
        }
      }

      /* Release our handle on the device */
      //uvc_close(devh);
      puts("Device closed");
    }

    /* Release the device descriptor */
    puts("RELEASING");
    //uvc_unref_device(dev);
  }

  /* Close the UVC context. This closes and cleans up any existing device handles,
   * and it closes the libusb context if one was not provided. */
  // uvc_exit(ctx);
  puts("UVC exited");

  return 0;
}