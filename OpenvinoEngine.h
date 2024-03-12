#pragma once
#ifndef __OPENVINOENGINE_H__
#define __OPENVINOENGINE_H__

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> 
#include <windows.h>



class OpenvinoEngine {
public:
	OpenvinoEngine() {
	}
	OpenvinoEngine(const std::string& xml, const std::string& device) : xml_(xml), device_(device)
	{
		compileModel();
		createInferRequest();
	}
	OpenvinoEngine(const std::string& xml, const std::string& device, const std::string& filename): xml_(xml), device_(device),filename_(filename)
	{
		SetConsoleOutputCP(CP_UTF8);

		// 处理路径分隔符（可根据需要添加）
		std::string processedFilename = filename;
		// 在此处添加路径处理代码（例如替换分隔符）

		std::ifstream file(processedFilename);
		if (!file) {
			std::cerr << "Error opening file: " << processedFilename << std::endl;
		}
		std::string line;
		if (file.is_open()) {
			while (std::getline(file, line)) {
				if (!line.empty()) {
					keys.push_back(line);
					//std::cout << line << std::endl; // 打印每一行的内容
				}
			}
			file.close();
		}
		else {
			std::cout << "无法打开文件" << std::endl;
		}
		if (file.bad()) {
			std::cerr << "Error occurred while reading file: " << processedFilename << std::endl;
		}
		//printf("load keys successed");
		compileModel();	
		file.close();
	}
	void compileModel() {
		// 设置日志级别
		//
		// core_.set_property(ov::log::level(ov::log::Level::TRACE));
		try {

		std::shared_ptr<ov::Model> model = core_.read_model(xml_);
		ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
		ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
		ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255., 255., 255. });
		ppp.input().model().set_layout("NCHW");
		ppp.output().tensor().set_element_type(ov::element::f32);

		model = ppp.build();
		//printf("build successed\n");

		//compiled_model_ = std::make_shared<ov::CompiledModel>(core_.compile_model(model, device_));
	
			compiled_model_ = std::make_shared<ov::CompiledModel>(core_.compile_model(model, device_));
		}
		catch (const ov::Exception& e) {
			std::cerr << "OpenVINO Exception: " << e.what() << std::endl;
			// 根据异常信息采取进一步的行动
		}
		catch (const std::exception& e) {
			std::cerr << "Standard Exception: " << e.what() << std::endl;
			// 处理其他类型的标准异常
		}
		catch (...) {
			std::cerr << "Unknown error occurred during model compilation." << std::endl;
			// 处理未知异常
		}


		//printf("load successed\n");
	}

	void createInferRequest() {
		infer_request_ = std::make_shared<ov::InferRequest>(compiled_model_->create_infer_request());
	}

	//virtual void loadModel(const std::string& modelPath) = 0;
	//virtual void preprocess(const cv::Mat& inputImage) = 0;
	//virtual void postprocess() = 0;

	std::vector <float*> inference(float* data, int n);

public:
	std::string xml_;
	std::string device_;
	std::string filename_;
	ov::Core core_;
	std::shared_ptr<ov::CompiledModel> compiled_model_;
	std::shared_ptr<ov::InferRequest> infer_request_;
	std::vector<std::string> keys;

};

#endif