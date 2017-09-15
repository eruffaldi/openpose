#include "utilities.hpp"


int getHeight(const std::string & resolution)
{
  std::string height = resolution.substr(resolution.find("x") + 1);
  return atoi(height.c_str());
}

int getWidth(const std::string & resolution)
{
  std::string width = resolution.substr(0,resolution.find("x"));
  return atoi(width.c_str());
}

int getInt(const std::string & s, const std::string c)
{
	std::string height = s.substr(s.find(c) + 1);
	return atoi(height.c_str());	
}

double getDouble(const std::string & s, const std::string c)
{
	std::string height = s.substr(s.find(c) + 1);
	return atof(height.c_str());	
}

//*Black Magic*/
constexpr unsigned int str2int(const char* str, int h)
{
  return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

const std::string getResolutionCode(const std::string resolution)
{
  switch(str2int(resolution.c_str()))
  {
    case str2int("672x376"):
      return "CAM_VGA";
    case str2int("1280x720"):
      return "CAM_HD";
    case str2int("1920x1080"):
      return "CAM_FHD";
    case str2int("2208x1242"):
      return "CAM_2K";
    default:
      return "NOT SUPPORTED RESOLUTION";
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

  return cv::Point2d((p3d[0]*fx/z+cx), (p3d[1]*fy/z +cy));
}
