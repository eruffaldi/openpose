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

	void triangulateCore(cv::Mat & cam0pnts, cv::Mat & cam1pnts, cv::Mat & finalpoints);

	void parseIntrinsicMatrix(const std::string path = "../settings/SN1499.conf");

	virtual cv::Mat triangulate();

	virtual void visualize(bool* keep_on);

	virtual void process(const cv::Mat & image);

	virtual void verify(const cv::Mat & pnts, bool* keep_on);


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
                                              : StereoPoseExtractor(argc,argv,resolution), filepath_(path), file_(path)
    {	


    	if(file_.is_open())
    	{
    		getline(file_,line_);
    		getline(file_,line_);
    	}
    	else
    	{
    		std::cout << "Could not open keypoints file!" << std::endl;
    		exit(-1);
    	}	
    }
                                        
	virtual void process(const cv::Mat & image);

	void getNextBlock(std::vector<std::vector<std::string>> & lines);

	virtual void visualize(bool * keep_on);

	virtual cv::Mat triangulate();

	const std::string filepath_;
	std::ifstream file_;
	std::string line_;

	std::vector<cv::Point2d> points_left_;
	std::vector<cv::Point2d> points_right_;

};