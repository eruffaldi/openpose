#include "stereoprocess.h"
#include "utilities.hpp"

void DisparityExtractor::getDisparity()
{ 
  cv::Mat grayleft,grayright;

  cv::cvtColor(imageleft_,grayleft, CV_BGR2GRAY);
  cv::cvtColor(imageright_,grayright, CV_BGR2GRAY);

  gpuleft_.upload(grayleft);
  gpuright_.upload(grayright);

  disparter_->compute(gpuleft_,gpuright_,disparity_);
}

/*
* x: point coordinate in pixel
* y: point coordinate in pixel
* d: disparity at point (x,y)
*/
/*
cv::Point3d DisparityExtractor::getPointFromDisp(double u, double v, double d)
{


  std::cout << "Reconstructing 3D point from " << u << " " << v << " " << d << std::endl;

  cv::Mat hom_point3D(4,1,CV_64FC1);
  cv::Mat hom_point2DD = (cv::Mat_<double>(4,1) << u, v, d, 1.0);

  hom_point3D = iP_ * hom_point2DD;

  double W = hom_point3D.at<double>(3,0);

  return cv::Point3d(hom_point3D.at<double>(0,0)/W,hom_point3D.at<double>(1,0)/W,hom_point3D.at<double>(2,0)/W);
}
*/

/*
* x: point coordinate in pixel
* y: point coordinate in pixel
* d: disparity at point (x,y)
*/
cv::Point3d DisparityExtractor::getPointFromDisp(double u, double v, double d)
{

  double f = cam_.intrinsics_left_.at<double>(0,0);
  double b = -cam_.ST_[0];

  double Z = (f * b)/d;
  double X = (u * Z)/f;
  double Y = (v * Z)/f;

  return cv::Point3d(X,Y,Z);

}

double DisparityExtractor::avgDisp(const cv::Mat & disp, int u, int v, int side)
{

  //TODO:check is not a border point
  double wlb,wub;
  double hlb,hub;

  wlb = std::max(0,u - side);
  hlb = std::max(0,v - side);

  wub = std::min(cam_.width_, u + side + 1);
  hub = std::min(cam_.height_,v + side + 1);

  double count = 0.0;
  double ret = 0.0;

  for (int i = wlb; i < wub; i++)
  {
    for (int j = hlb; j < hub; j++)
    { 

      ret = ret + (double)disp.at<uint8_t>(i,j);
     if((double)disp.at<uint8_t>(i,j) > 0.0)
     {
        count ++;   
     }
    }
  }

  return ret/count;
}

double DisparityExtractor::triangulate(cv::Mat & output) 
{

  getDisparity();

  cv::Mat disp,disp8;
  cv::Mat cam0pnts,cam1pnts;

  disparity_.download(disp);  
  getPoints(cam0pnts,cam1pnts);
  filterVisible(cam0pnts,cam1pnts,cam0pnts,cam1pnts);

  output = cv::Mat(1,cam0pnts.cols,CV_64FC3);

  for(int i=0; i < cam0pnts.cols; i++)
  {

    cv::Vec2d p = cam0pnts.at<cv::Vec2d>(0,i);
    double dispatpoint = (double)disp.at<uint8_t>(cvRound(p[0]),cvRound(p[1]));
    double disparity_point = avgDisp(disp,cvRound(p[0]),cvRound(p[1]),3);
    cv::Point3d p3 = getPointFromDisp(p[0],p[1],dispatpoint);
    output.at<cv::Point3d>(0,i) = p3;
  }

  std::cout << "point: " << output.at<cv::Point3d>(0,0) << std::endl;

  return getRMS(cam0pnts, output);
}


static void saveXYZ(const char* filename, const cv::Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}


void DisparityExtractor::verify(const cv::Mat & pnts, bool* keep_on)
{ 

  if(pnts.empty())
  {
    return;
  }
  
  std::vector<cv::Point2d> points2D(pnts.cols);
  std::vector<cv::Vec4d> point2DD(pnts.cols);

  std::cout << "Projecting points:\n " << pnts << std::endl;

  for (unsigned int i = 0; i < pnts.cols; i++)
  {

    cv::Vec3d curpoint = pnts.at<cv::Vec3d>(0,i);
    cv::Mat hom3D = (cv::Mat_<double>(4,1) << curpoint[0], curpoint[1], curpoint[2], 1.0);

    cv::Mat hom2DD = P_ * hom3D;

    double W = hom2DD.at<double>(0,3);


    cv::Point2d point2D(hom2DD.at<double>(0,0)/W, hom2DD.at<double>(0,1)/W);
    points2D[i] = point2D;
   
  } 
  //TODO: write circles in projected points
  cv::Mat verification = imageright_.clone();
  for (auto & c : points2D)
  {
    cv::circle(verification,c,4,255,2);
  }


  cv::namedWindow("Verification", CV_WINDOW_AUTOSIZE);
  cv::imshow("Verification", verification);
  
  int k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
  if (k == 's')
  {  
    cv::Mat disp;
    disparity_.download(disp);  
    std::cout << "storing the cloud " << std::endl;
    cv::Mat xyz;
    cv::reprojectImageTo3D(disp, xyz, Q_, true);
    saveXYZ("../data/3Dpoints.pcd", xyz);
  }
}

void DisparityExtractor::extract(const cv::Mat & image)
{

  cur_frame_ ++;
  splitVertically(image, imageleft_, imageright_);

  cv::Mat left_undist, right_undist;

  //cv::remap(imageleft_, left_undist, map11_, map12_, cv::INTER_LINEAR);
  //cv::remap(imageright_, right_undist, map21_, map22_, cv::INTER_LINEAR);

  //cv::undistort(imageleft_, left_undist, cam_.intrinsics_left_, cam_.dist_left_);
  //cv::undistort(imageright_, right_undist, cam_.intrinsics_right_, cam_.dist_right_);

  //imageleft_ = left_undist;
  //imageright_ = right_undist;

  //cv::pyrDown(imageleft_, imageleft_);
  //cv::pyrDown(imageright_, imageright_);
}

void DisparityExtractor::visualize(bool * keep_on)
{ 

  cv::Mat disp,disp8;
  cv::Mat cam0pnts,cam1pnts;

  disparity_.download(disp); 

  //std::cout << "disparity " << std::endl;
  //std::cout << disp << std::endl;

  //cv::hconcat(outputImageL_, outputImageR_, sidebyside_out);
  if(!disp.empty())
  {
    cv::namedWindow("Side By Side", CV_WINDOW_AUTOSIZE);
    cv::imshow("Side By Side", disp);
  }

  int k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
}