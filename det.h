#pragma once
#ifndef __DET_H__
#define __DET_H__

#include "OpenvinoEngine.h"
#include "common.h"
#include "clipper.h"

class Det : public OpenvinoEngine {
public:
	Det(const std::string& model_xml, const std::string& device)
		: OpenvinoEngine(model_xml, device) {}
 
	Det():OpenvinoEngine(){}
	std::vector<cv::Mat> inference(cv::Mat img, int img_len);
	std::vector<TextBox> inference2(cv::Mat img, int img_len, float boxScoreThresh,float boxThresh);

	std::vector<TextBox> findRsBoxes(const cv::Mat& fMapMat, const cv::Mat& norfMapMat,
		const float boxScoreThresh, const float unClipRatio);
	void drawBoxesOnImage(cv::Mat& img, const std::vector<TextBox>& textBoxes);
public:
	float data[3 * 960 * 960];
};

#endif