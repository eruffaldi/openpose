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

	virtual void getPoints(cv::Mat &, cv::Mat &);

	virtual double triangulate(cv::Mat &);

	virtual void visualize(bool* keep_on);

	virtual void process();

	virtual void extract(const cv::Mat &);

	virtual void verify(const cv::Mat & pnts, bool* keep_on);

	virtual double getRMS(const cv::Mat &, const cv::Mat & pnts3D);

	virtual double go(const cv::Mat & image, const bool verify, cv::Mat &, bool* keep_on);


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

struct DisparityExtractor : StereoPoseExtractor {

	DisparityExtractor(int argc, char **argv, const std::string resolution) : StereoPoseExtractor(argc,argv,resolution){

		double f = cam_.intrinsics_left_.at<double>(0,0);
		double cx = cam_.intrinsics_left_.at<double>(0,2);
		double cy = cam_.intrinsics_left_.at<double>(1,2);
		double B = cam_.ST_[0];

		//TODO: build the Q matrix
		cv::Mat K4 = (cv::Mat_<double>(4,4) << f, 0.0, 0.0, cx, 0.0, f, 0.0, cy, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0);
		cv::Mat RT = cv::Mat::eye(4,4,CV_64FC1);
		RT.at<double>(0,3) = B;

		P_ = K4 * RT;
		iP_ = P_.inv();

	    disparter_->setPreFilterCap(31);
	    disparter_->setBlockSize(9);
	    disparter_->setMinDisparity(0);
	    disparter_->setTextureThreshold(10);
	    disparter_->setUniquenessRatio(15);
	    disparter_->setSpeckleWindowSize(100);
	    disparter_->setSpeckleRange(32);
	    disparter_->setDisp12MaxDiff(1);

	    cv::Mat R1,R2,P1,P2;

	    cv::stereoRectify(cam_.intrinsics_left_,cam_.dist_left_,cam_.intrinsics_right_,cam_.dist_right_,cv::Size(cam_.width_,cam_.height_),cam_.SR_, cam_.ST_,
	                      R1,R2,P1,P2,Q_);

	}

	void getDisparity();

	cv::Point3d getPointFromDisp(double u, double v, double d);

	double avgDisp(const cv::Mat & disp, int u, int v, int side = 5);

	void verifyD(const cv::Mat & pnts, bool* keep_on);

	virtual void extract(const cv::Mat & image);

	virtual void verify(const cv::Mat & pnts, bool* keep_on);

	virtual double triangulate(cv::Mat & output); 

	cv::cuda::GpuMat disparity_;
	cv::cuda::GpuMat gpuleft_,gpuright_;

	cv::Mat P_;
	cv::Mat iP_;
	cv::Mat Q_;

	cv::Ptr<cv::cuda::StereoBM> disparter_ = cv::cuda::createStereoBM(16,9);

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

	virtual void visualize(bool * keep_on);

	virtual void getPoints(cv::Mat & outputL, cv::Mat & outputR);

	void getNextBlock(std::vector<std::vector<std::string>> & lines);

	const std::string filepath_;
	std::ifstream file_;
	std::string line_;

	std::vector<cv::Point2d> points_left_;
	std::vector<cv::Point2d> points_right_;

};