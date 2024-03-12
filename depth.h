#pragma once
#ifndef __DEPTH_H__
#define __DEPTH_H__

#include "OpenvinoEngine.h"

struct PaddedResizedImage {
	cv::Mat image; // 调整大小后的图像
	int originalWidth; // 原始图像宽度
	int originalHeight; // 原始图像高度
	int topPadding; // 顶部填充
	int bottomPadding; // 底部填充
	int leftPadding; // 左侧填充
	int rightPadding; // 右侧填充
};

class Depth : public OpenvinoEngine {
public:
	Depth(const std::string& model_xml, const std::string& device)
		: OpenvinoEngine(model_xml, device) {}

	Depth() :OpenvinoEngine() {}
    PaddedResizedImage paddingAndResize(const cv::Mat& img, int img_len) {
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

        // 返回结构体，包含处理后的图像和恢复信息
        return { resizedImg, img.cols, img.rows, top, bottom, left, right };
    }
    cv::Mat restoreOriginalImage(const PaddedResizedImage& paddedImage) {
        // 首先，将图像调整回填充前的尺寸
        cv::Mat beforePadding;
        cv::resize(paddedImage.image, beforePadding, cv::Size(paddedImage.originalWidth + paddedImage.leftPadding + paddedImage.rightPadding, paddedImage.originalHeight + paddedImage.topPadding + paddedImage.bottomPadding));

        // 然后，裁剪掉填充的部分
        cv::Rect roi(paddedImage.leftPadding, paddedImage.topPadding, paddedImage.originalWidth, paddedImage.originalHeight);
        cv::Mat originalImage = beforePadding(roi);

        return originalImage;
    }
	cv::Mat inference(cv::Mat img, int h,int w);
  
public:
 };

#endif