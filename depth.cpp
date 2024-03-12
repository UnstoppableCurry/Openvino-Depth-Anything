#define NOMINMAX
#include "depth.h"
#include <iostream>


/*
PaddedResizedImage  Depth::paddingAndResize(const cv::Mat& img, int img_len) {
	// 确定需要填充的边长以使图像成为1:1的纵横比
	int maxSide = std::max(img.cols, img.rows);

	// 计算水平和垂直方向上的填充大小
	int deltaWidth = maxSide - img.cols;
	int deltaHeight = maxSide - img.rows;
	int top = deltaHeight / 2;
	int bottom = deltaHeight - top;
	int left = deltaWidth / 2;
	int right = deltaWidth - left;

	// 填充图像
	cv::Mat squareImg;
	cv::copyMakeBorder(img, squareImg, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	// 调整填充后的图像大小到 img_len x img_len
	cv::Mat resizedImg;
	cv::resize(squareImg, resizedImg, cv::Size(img_len, img_len));

	//return resizedImg;
	return { resizedImg, img.cols, img.rows, top, bottom, left, right };

}

*/
cv::Mat Depth::inference(cv::Mat img,int h,int w)
{
	PaddedResizedImage result=paddingAndResize(img, w);
	cv::Mat pad = result.image;

	float* data = (float*)pad.data;

	auto input_port = compiled_model_->input();
	ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), data);

	infer_request_->set_input_tensor(input_tensor);
	infer_request_->infer();
	auto output_tensor = infer_request_->get_output_tensor(); //output(n)
	auto output_shape = output_tensor.get_shape();
	float* detections = output_tensor.data<float>();
	output_shape[0] = h;
	output_shape[1] = w;
	// 创建一个与推理结果形状相匹配的 cv::Mat 对象
	cv::Mat resultMat(output_shape[0], output_shape[1], CV_32FC1);
	std::memcpy(resultMat.data, detections, output_shape[0] * output_shape[1] * sizeof(float));
	cv::Mat color_map_ = cv::Mat(h, w, CV_8UC3);

	cv::normalize(resultMat, resultMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::applyColorMap(resultMat, color_map_, cv::ColormapTypes::COLORMAP_INFERNO);

	// 将数据处理为 uint8 并缩放到 [0, 255]
	//cv::Mat resultMatUint8;
	//resultMat.convertTo(resultMatUint8, CV_8U, 255.0);
	result.image = color_map_;
	cv::Mat restoredImg = restoreOriginalImage(result);

	return restoredImg;

}
static int draw_fps(cv::Mat& rgb)
{
	// resolve moving average
	float avg_fps = 0.f;
	{
		static int64 t0 = 0;
		static float fps_history[10] = { 0.f };

		int64 t1 = cv::getTickCount();
		if (t0 == 0)
		{
			t0 = t1;
			return 0;
		}

		float fps = cv::getTickFrequency() / (t1 - t0);
		t0 = t1;

		for (int i = 9; i >= 1; i--)
		{
			fps_history[i] = fps_history[i - 1];
		}
		fps_history[0] = fps;

		if (fps_history[9] == 0.f)
		{
			return 0;
		}

		for (int i = 0; i < 10; i++)
		{
			avg_fps += fps_history[i];
		}
		avg_fps /= 10.f;
	}

	char text[32];
	sprintf(text, "FPS=%.2f", avg_fps);

	int baseLine = 0;
	cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

	int y = 0;
	int x = rgb.cols - label_size.width;

	cv::rectangle(rgb, cv::Point(x, y), cv::Point(x + label_size.width, y + label_size.height + baseLine), cv::Scalar(255, 255, 255), -1);

	cv::putText(rgb, text, cv::Point(x, y + label_size.height),
		cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

	return 0;
}
int maindepth() {
	const std::string path = "H:\\vsworkspace\\AperturePluginInferCPP\\x64\\Release\\mo\\depth\\depth.xml";
	//std::string path = "H:\\\\vsworkspace\\\\ocr_sdk\\\\x64\\\\Release\\\\mo\\\\det_16\\\\det.xml";

	std::string imgpath = "H:\\vsworkspace\\ocr_sdk\\x64\\Release\\img\\12.jpg";
	printf("start \n");
	Depth* depth = new Depth(path, "CPU");
	cv::Mat frame;   //声明一个保存图像的类
	cv::VideoCapture video;   //用VideoCapture来读取摄像头
	video.open(0);   //括号的0表示使用电脑自带的摄像头

	while (1)   //（读取成功，使用循环语句将视频一帧一帧地展示出来）
	{
		video >> frame;
		cv::Mat result = depth->inference(frame, 518, 518);
		draw_fps(frame); // 在帧上绘制FPS信息
		cv::imshow("frame", frame);
		cv::imshow("result", result);
		cv::waitKey(1);

	}


	delete depth;
	printf("end \n");
	return 0;

}