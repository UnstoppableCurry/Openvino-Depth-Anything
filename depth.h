#pragma once
#ifndef __DEPTH_H__
#define __DEPTH_H__

#include "OpenvinoEngine.h"

struct PaddedResizedImage {
	cv::Mat image; // ������С���ͼ��
	int originalWidth; // ԭʼͼ����
	int originalHeight; // ԭʼͼ��߶�
	int topPadding; // �������
	int bottomPadding; // �ײ����
	int leftPadding; // ������
	int rightPadding; // �Ҳ����
};

class Depth : public OpenvinoEngine {
public:
	Depth(const std::string& model_xml, const std::string& device)
		: OpenvinoEngine(model_xml, device) {}

	Depth() :OpenvinoEngine() {}
    PaddedResizedImage paddingAndResize(const cv::Mat& img, int img_len) {
        // ȷ����Ҫ���ı߳���ʹͼ���Ϊ1:1���ݺ��
        int maxSide = std::max(img.cols, img.rows);

        // ����ˮƽ�ʹ�ֱ�����ϵ�����С
        int deltaWidth = maxSide - img.cols;
        int deltaHeight = maxSide - img.rows;
        int top = deltaHeight / 2;
        int bottom = deltaHeight - top;
        int left = deltaWidth / 2;
        int right = deltaWidth - left;

        // ���ͼ��
        cv::Mat squareImg;
        cv::copyMakeBorder(img, squareImg, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        // ���������ͼ���С�� img_len x img_len
        cv::Mat resizedImg;
        cv::resize(squareImg, resizedImg, cv::Size(img_len, img_len));

        // ���ؽṹ�壬����������ͼ��ͻָ���Ϣ
        return { resizedImg, img.cols, img.rows, top, bottom, left, right };
    }
    cv::Mat restoreOriginalImage(const PaddedResizedImage& paddedImage) {
        // ���ȣ���ͼ����������ǰ�ĳߴ�
        cv::Mat beforePadding;
        cv::resize(paddedImage.image, beforePadding, cv::Size(paddedImage.originalWidth + paddedImage.leftPadding + paddedImage.rightPadding, paddedImage.originalHeight + paddedImage.topPadding + paddedImage.bottomPadding));

        // Ȼ�󣬲ü������Ĳ���
        cv::Rect roi(paddedImage.leftPadding, paddedImage.topPadding, paddedImage.originalWidth, paddedImage.originalHeight);
        cv::Mat originalImage = beforePadding(roi);

        return originalImage;
    }
	cv::Mat inference(cv::Mat img, int h,int w);
  
public:
 };

#endif