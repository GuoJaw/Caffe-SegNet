/*
 *
 *  Created on: Jan 21, 2017
 *      Author: Timo Sämann
 *
 *  The basis for the creation of this script was the classification.cpp example (caffe/examples/cpp_classification/classification.cpp)
 *
 *  This script visualize the semantic segmentation for your input image.
 *
 *  To compile this script you can use a IDE like Eclipse. To include Caffe and OpenCV in Eclipse please refer to
 *  http://tzutalin.blogspot.de/2015/05/caffe-on-ubuntu-eclipse-cc.html
 *  and http://rodrigoberriel.com/2014/10/using-opencv-3-0-0-with-eclipse/ , respectively
 *
 *
 */



#define USE_OPENCV 1
#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
//#include <chrono> //Just for time measurement. This library requires compiler and library support for the ISO C++ 2011 standard. This support is currently experimental in Caffe, and must be enabled with the -std=c++11 or -std=gnu++11 compiler options.


#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace cv;

#define CLASS_PEOPLE 15

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file);


  void Predict(const cv::Mat& img, string LUT_file);

 private:
  void SetMean(const string& mean_file);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  void Visualization(Blob<float>* output_layer, string LUT_file);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;

};

Classifier::Classifier(const string& model_file,
                       const string& trained_file) {


  Caffe::set_mode(Caffe::GPU);

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}


void Classifier::Predict(const cv::Mat& img, string LUT_file) {  //LUT_file标签图像，用于LUT
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);


  struct timeval time;
  gettimeofday(&time, NULL); // Start Time
  long totalTime = (time.tv_sec * 1000) + (time.tv_usec / 1000);
  //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //Just for time measurement

  net_->Forward();

  gettimeofday(&time, NULL);  //END-TIME
  totalTime = (((time.tv_sec * 1000) + (time.tv_usec / 1000)) - totalTime);
  std::cout << "Processing time = " << totalTime << " ms" << std::endl;

      /* Copy the output layer to a std::vector */
      Blob<float>* output_layer = net_->output_blobs()[0];

      //获得只有人的灰度图像
     // 仿照博客: caffe之特征图可视化及特征提取 https://blog.csdn.net/zxj942405301/article/details/71195267
     //将测试图片经过最后一个卷积层的特征图可视化
     //这一层的名字叫做 prob
     string blobName="prob";
     assert(net_->has_blob(blobName)); //为免出错，我们必须断言，网络中确实有名字为blobName的特征图
     boost::shared_ptr<Blob<float> > conv1Blob=net_->blob_by_name(blobName);
     //为了可视化，仍然需要归一化到0~255
     //下面的代码，跟上一篇博客中是一样的
     float maxValue=-10000000,minValue=10000000;
     const float* tmpValue=conv1Blob->cpu_data();
     for(int i=0;i<conv1Blob->count();i++)
     {
         maxValue=std::max(maxValue,tmpValue[i]);
         minValue=std::min(minValue,tmpValue[i]);
     }
     int width=conv1Blob->shape(2); //响应图的宽度
     int height=conv1Blob->shape(3); //响应图的高度
     //获取第15类 "人"
     Mat img_i(width,height,CV_8UC1);
     for(int i=0;i<height;i++)
     {
         for(int j=0;j<width;j++)
         {
             float value=conv1Blob->data_at(0,CLASS_PEOPLE,i,j);
             img_i.at<uchar>(i,j)=(value-minValue)/(maxValue-minValue)*255;
         }
     }
     //转化成与原图片相同的格式
     resize(img_i,img_i,Size(640,480));
     //将结果输出
     cv::Mat img_segment = img_i.clone();
     cv::imshow("img_segment",img_segment);

     //Visualization(output_layer, LUT_file);
}


void Classifier::Visualization(Blob<float>* output_layer, string LUT_file) {  //可视化语义分割后的图像

  std::cout << "output_blob(n,c,h,w) = " << output_layer->num() << ", " << output_layer->channels() << ", "    //output_layer->channels() 21类别
			  << output_layer->height() << ", " << output_layer->width() << std::endl;

  cv::Mat merged_output_image = cv::Mat(output_layer->height(), output_layer->width(), CV_32F, const_cast<float *>(output_layer->cpu_data()));

  //merged_output_image = merged_output_image/255.0;

  merged_output_image.convertTo(merged_output_image, CV_8U);
  cv::cvtColor(merged_output_image.clone(), merged_output_image, CV_GRAY2BGR); //merged_output_image创建的原图

  cv::Mat label_colours = cv::imread(LUT_file,1); ///label_colours颜色标签图像
  cv::Mat output_image;


  //对于一个给定的值，将其替换成其他的值是一个很常见的操作
  LUT(merged_output_image, label_colours, output_image); //output_image输出图像

  cv::namedWindow("Display window",0);
  cv::imshow( "Display window", output_image);
  
 

//查找轮廓
  cv::cvtColor( output_image, output_image, CV_RGB2GRAY );  //灰度化处理
  cv::threshold(output_image, output_image, 50, 255, cv::THRESH_BINARY);  //阈值化处理
  std::vector< std::vector< cv::Point> > contours;  
  cv::findContours(  
    output_image,  
    contours,  
    cv::noArray(),  
    cv::RETR_LIST,  
    cv::CHAIN_APPROX_SIMPLE  
    );  
  output_image = cv::Scalar::all(0);  


//筛选剔除掉面积小于100的轮廓
  for(int i=0;i<contours.size();i++)
  {
       double length = cv::arcLength(contours[i],true);
       if(length > 400)
          cv::drawContours(output_image, contours, i, cv::Scalar::all(255));  //画上轮廓
  }
  cv::namedWindow("Contours",0);
  cv::imshow("Contours", output_image);  

}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv)
{
  string model_file="../model/segnet_pascal.prototxt"; //.prototxt
  string trained_file = "../model/segnet_pascal.caffemodel"; //.caffemodel

  Classifier classifier(model_file, trained_file);

  string LUT_file = "../model/sun.png";

  cv::VideoCapture cap(0);
  while(1)
  {
	cv::Mat img;   cap>>img;

	CHECK(!img.empty()) << "Unable to decode image ";
	cv::Mat prediction;

	classifier.Predict(img, LUT_file);
    char c = cv::waitKey(1);
	if(c == 27)
		break;
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV


