#pragma once
#include "libuvc/libuvc.h"
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include "utilities.hpp"
#include "stereo_cam.h"
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging

extern bool keep_on;


void splitVertically(const cv::Mat & input, cv::Mat & outputleft, cv::Mat & outputright);
std::vector<std::string> CSVTokenize(std::string kpl_str);
void emitCSV(std::ofstream & outputfile, std::string & kp_str, const op::Array<float> & poseKeypoints, int camera);


struct StereoPoseExtractor {

	StereoPoseExtractor(int argc, char **argv, const std::string resolution);

	void init();

	void destroy();

	void process(const cv::Mat & image);

	void visualize(bool* keep_on);

	void parseIntrinsicMatrix(const std::string path = "../settings/SN1499.conf");

	virtual cv::Mat triangulate();

	void verify(const cv::Mat & pnts, bool* keep_on);

	std::vector<cv::Point3f> triangulate(const std::string &);

	op::CvMatToOpInput *cvMatToOpInput_;
	op::CvMatToOpOutput *cvMatToOpOutput_;
	op::PoseExtractorCaffe *poseExtractorCaffeL_;
	op::PoseRenderer *poseRendererL_;
	op::OpOutputToCvMat *opOutputToCvMatL_;
	op::OpOutputToCvMat *opOutputToCvMatR_;

	op::Array<float> poseKeypointsL_;
	op::Array<float> poseKeypointsR_;

	bool inited_;

	cv::VideoWriter outputVideo_; 
	std::ofstream outputfile_;   

	int cur_frame_;	

	cv::Mat imageleft_;
	cv::Mat imageright_;
	cv::Mat outputImageR_;
	cv::Mat outputImageL_;

	StereoCamera cam_;
};

struct PoseExtractorFromFile : StereoPoseExtractor {

	PoseExtractorFromFile(int argc, char **argv, const std::string resolution, const std::string path) 
                                              : StereoPoseExtractor(argc,argv,resolution), filepath_(path){}
                                        
	cv::Mat triangulate();

	const std::string filepath_;

};