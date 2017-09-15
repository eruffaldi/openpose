#include "stereopose.h"


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
DEFINE_string(model_folder,             "/home/lando/projects/openpose_stereo/openpose/models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased,"
                                                        " the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect"
                                                        " ratio possible to the images or videos to be processed. E.g. the default `656x368` is"
                                                        " optimal for 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) videos.");

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
DEFINE_string(write_video,              "",             "Full file path to write rendered frames in motion JPEG video format.");

DEFINE_string(write_keypoint,           "",             "Full file path to write people body pose keypoints data. Only CSV format supported");  

DEFINE_string(keypoint_file,            "",             "Full file path to read keypoint file. Used to get 3D keypoints from file instead of video");


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

/* Format string in CSV format 
 *
 */
std::vector<std::string> CSVTokenize(std::string kpl_str)
{
  kpl_str.erase(0, kpl_str.find("\n") + 1);
  std::replace(kpl_str.begin(), kpl_str.end(), '\n', ' ');

  std::vector<std::string> vec;
  std::istringstream iss(kpl_str);
  copy(std::istream_iterator<std::string>(iss),std::istream_iterator<std::string>(),back_inserter(vec));

  return vec;
}

void emitCSV(std::ofstream & outputfile, std::string & kp_str, const op::Array<float> & poseKeypoints, int camera, int cur_frame)
{ 
   std::vector<std::string> tokens = CSVTokenize(kp_str);

   std::cout << "number of strings " << tokens.size() << std::endl;

   //if no person detected, output 54 zeros
   if (tokens.size() == 0)
   {
     outputfile << camera << " " << cur_frame << " " << 0 << " ";
     for (int j = 0; j < 54; j++)
     {
       outputfile << 0.000 << " ";
     }

     outputfile << '\n';
   }

  for (int i = 0; i < poseKeypoints.getVolume(); i += 54)
   {
     outputfile << camera << " " << cur_frame << " " << i/54 << " ";
     for (int j = 0; j < 54; j++)
     {
       outputfile << tokens[i+j] << " ";
     }

     outputfile << '\n';
   }  
}

StereoPoseExtractor::StereoPoseExtractor(int argc, char **argv, const std::string resolution) : cam_(resolution)
{  

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  inited_ = false;
  cur_frame_ = 0;

  if (FLAGS_write_video != "")
  { 
    cv::Size S = cv::Size(2560, 720);
    outputVideo_.open(FLAGS_write_video, CV_FOURCC('M','J','P','G'), 7, S, true);
    if (!outputVideo_.isOpened())
    {
        std::cout  << "Could not open the output video for write: " << std::endl;
        exit(-1);
    }
  }

  if (FLAGS_write_keypoint != "")
  {
    outputfile_.open(FLAGS_write_keypoint);
    //TODO:write header of outputfilef
    outputfile_ << "camera frame subject ";
    for (int i = 0; i < 18; i++)
    {
      outputfile_ << "p" << i << "x" << " p" << i << "y" << " p" << i << "conf ";
    }
    outputfile_ << "\n";
  }

  // Step 2 - Read Google flags (user defined configuration)
  // outputSize
  const auto outputSize = op::flagsToPoint(cam_.resolution_, "1280x720");
  // netInputSize
  const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "656x368");
  // netOutputSize
  const auto netOutputSize = netInputSize;
  // poseModel
  const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);

  // Step 3 - Initialize all required classes
  cvMatToOpInput_ = new op::CvMatToOpInput{netInputSize, FLAGS_scale_number, (float)FLAGS_scale_gap};
  cvMatToOpOutput_ = new op::CvMatToOpOutput{outputSize};
  poseExtractorCaffeL_ = new op::PoseExtractorCaffe{netInputSize, netOutputSize, outputSize, FLAGS_scale_number, poseModel,
                                                FLAGS_model_folder, FLAGS_num_gpu_start};
  poseRendererL_ = new op::PoseRenderer{netOutputSize, outputSize, poseModel, nullptr, (float)FLAGS_render_threshold,
                                    !FLAGS_disable_blending, (float)FLAGS_alpha_pose};
  opOutputToCvMatL_ = new op::OpOutputToCvMat{outputSize};
  opOutputToCvMatR_ = new op::OpOutputToCvMat{outputSize};
}

void StereoPoseExtractor::init()
{
  if (inited_ == false)
  {
    poseExtractorCaffeL_->initializationOnThread();
    poseRendererL_->initializationOnThread();
    inited_ = true;
  }
}

void StereoPoseExtractor::destroy()
{
  outputfile_.close();
}

void StereoPoseExtractor::process(const cv::Mat & image)
{

  cur_frame_ ++;

  //Split image  vertically into 2 even parts
  splitVertically(image, imageleft_, imageright_);

  op::Array<float> netInputArrayL;
  op::Array<float> netInputArrayR;

  op::Array<float> outputArrayL;
  op::Array<float> outputArrayR;

  std::vector<float> scaleRatiosL;
  std::vector<float> scaleRatiosR;

  double scaleInputToOutputL;
  double scaleInputToOutputR;

  std::tie(netInputArrayL, scaleRatiosL) = cvMatToOpInput_->format(imageleft_);
  std::tie(scaleInputToOutputL, outputArrayL) = cvMatToOpOutput_->format(imageleft_);
  std::tie(netInputArrayR, scaleRatiosR) = cvMatToOpInput_->format(imageright_);
  std::tie(scaleInputToOutputR, outputArrayR) = cvMatToOpOutput_->format(imageright_);

  // Step 3 - Estimate poseKeypoints
  poseExtractorCaffeL_->forwardPass(netInputArrayL, {imageleft_.cols, imageleft_.rows}, scaleRatiosL);
  poseKeypointsL_ = poseExtractorCaffeL_->getPoseKeypoints();

  poseExtractorCaffeL_->forwardPass(netInputArrayR, {imageright_.cols, imageright_.rows}, scaleRatiosR);
  poseKeypointsR_ = poseExtractorCaffeL_->getPoseKeypoints();

  std::string kpl_str = poseKeypointsL_.toString();
  std::string kpr_str = poseKeypointsR_.toString();

  // Step 4 - Render poseKeypoints
  poseRendererL_->renderPose(outputArrayL, poseKeypointsL_);
  poseRendererL_->renderPose(outputArrayR, poseKeypointsR_);    
  
  // Step 5 - OpenPose output format to cv::Mat
  outputImageL_ = opOutputToCvMatL_->formatToCvMat(outputArrayL);
  outputImageR_ = opOutputToCvMatL_->formatToCvMat(outputArrayR);

  if( FLAGS_write_video != "")
  { 
    cv::Mat sidebyside_in;
    cv::hconcat(imageleft_, imageright_, sidebyside_in);
    outputVideo_ << sidebyside_in;
  }

  if( FLAGS_write_keypoint != "")
  {
    emitCSV(outputfile_, kpl_str, poseKeypointsL_, 0, cur_frame_);
    emitCSV(outputfile_, kpr_str, poseKeypointsR_, 1, cur_frame_);
  }
}

/*
*TODO: check implementation on openpose library. There exists for sure.
*/
void fill2DMatrix(const op::Array<float> & keypoints, cv::Mat & campnts)
{

  double x = 0.0;
  double y = 0.0;

  //Ugliest AND SLOWEST
  std::vector<std::string> spoints = CSVTokenize(keypoints.toString());

  int people = keypoints.getVolume()/54;

  campnts = cv::Mat(1,people*18,CV_64FC2);

  for (int i = 0; i < 54 * people; i += 3)
  {
    x = atof(spoints[i].c_str());
    y = atof(spoints[i+1].c_str());
    cv::Vec2d elem(x,y);
    campnts.at<cv::Vec2d>(0,i/3) = elem;
  }
}


void filterVisible(const cv::Mat & pntsL, const cv::Mat & pntsR, cv::Mat & nzL, cv::Mat & nzR)
{ 
  cv::Vec2d pl;
  cv::Vec2d pr;
  cv::Vec2d zerov(0.0,0.0);

  std::vector<cv::Vec2d> pntsl;
  std::vector<cv::Vec2d> pntsr;

  for (int i = 0; i < pntsL.cols; i++)
  {
    pl = pntsL.at<cv::Vec2d>(0,i);
    pr = pntsR.at<cv::Vec2d>(0,i);


    if (pl != zerov && pr != zerov)
    {
      pntsl.push_back(pl);
      pntsr.push_back(pr);
    }

  }

  nzL = cv::Mat(1,pntsl.size(),CV_64FC2);
  nzR = cv::Mat(1,pntsr.size(),CV_64FC2);

  for (int i = 0; i < pntsl.size(); i++)
  {
    nzL.at<cv::Vec2d>(0,i) = pntsl[i];
    nzR.at<cv::Vec2d>(0,i) = pntsr[i];
  }
}


cv::Point2d project(const cv::Mat & intrinsics, const cv::Vec3d & p3d)
{   

  double z = p3d[2];
  //double z = 1.0;

  double fx = intrinsics.at<double>(0,0);
  double fy = intrinsics.at<double>(1,1);
  double cx = intrinsics.at<double>(0,2);
  double cy = intrinsics.at<double>(1,2);

  return cv::Point2d((p3d[0]*fx/z+cx)/2, p3d[1]*fy/z +cy);
}

cv::Mat StereoPoseExtractor::triangulate()
{

  //I can take all the points negleting if they belong to a specific person 
  //how can I know if the points belong to the same person? 
  int N = 0;
  cv::Mat cam0pnts;
  cv::Mat cam1pnts;
  cv::Mat nz_cam0pnts;
  cv::Mat nz_cam1pnts;

  if (poseKeypointsL_.empty() || poseKeypointsR_.empty())
  {
    return cv::Mat();
  }


  fill2DMatrix(poseKeypointsL_, cam0pnts);
  fill2DMatrix(poseKeypointsR_, cam1pnts);

  //TODO: check the numeber of detected people is the same, otherwise PROBLEM
  if(cam0pnts.cols != cam1pnts.cols)
  {
    std::cout << "number of detectd people differ" << std::endl;
    return cv::Mat();
  }

  N = nz_cam1pnts.cols;

  //Undistort points
  cv::Mat cam0pnts_undist(1,N,CV_64FC2);
  cv::Mat cam1pnts_undist(1,N,CV_64FC2);

  //If zeros in at least one: remove both 
  filterVisible(cam0pnts, cam1pnts, cam0pnts, cam1pnts);

  N = cam0pnts.cols;
  cv::Mat pnts3d(1,N,CV_64FC4);

  cv::Mat R1,R2,P1,P2,Q;
  /*Computes rectification transforms for each head of a calibrated stereo camera*/
  cv::stereoRectify(cam_.intrinsics_left_, cam_.dist_left_, cam_.intrinsics_right_, cam_.dist_right_, cv::Size(cam_.width_,cam_.height_), cam_.SR_,cam_.ST_, R1, R2, P1, P2, Q);

  cv::undistortPoints(cam0pnts, cam0pnts_undist, cam_.intrinsics_left_, cam_.dist_left_, R1, P1);
  cv::undistortPoints(cam1pnts, cam1pnts_undist, cam_.intrinsics_right_, cam_.dist_right_, R2, P2);

  cv::triangulatePoints(P1, P2, cam0pnts_undist, cam1pnts_undist, pnts3d);

  std::cout << "4D points " << std::endl;
  std::cout << pnts3d << std::endl;

  cv::Mat finalpoints(1,N,CV_64FC3);

  for (int i = 0; i < N; i++)
  { 
    cv::Vec4d cur = pnts3d.col(i);
    cv::Vec3d p3d(cur[0]/cur[3], cur[1]/cur[3],cur[2]/cur[3]);
    finalpoints.at<cv::Vec3d>(0,i) = p3d;
  }

  std::cout << "Triangulated point " << std::endl;
  std::cout << finalpoints.col(0) << std::endl;
  std::cout << "Projected point" << std::endl;
  cv::Vec3d prova = finalpoints.at<cv::Vec3d>(0,0);
  std::cout << project(cam_.intrinsics_left_, prova) << std::endl;

  return finalpoints;

}

std::vector<cv::Point3f> StereoPoseExtractor::triangulate(const std::string & filepath)
{
  std::string keypointpath = FLAGS_keypoint_file;
}

void StereoPoseExtractor::visualize(bool * keep_on)
{
  //TODO: make a video with 2 frame side by side
  cv::Mat sidebyside_out;
  cv::hconcat(outputImageL_, outputImageR_, sidebyside_out);

  cv::namedWindow("Side By Side", CV_WINDOW_AUTOSIZE);
  cv::imshow("Side By Side", sidebyside_out);

  int k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
}