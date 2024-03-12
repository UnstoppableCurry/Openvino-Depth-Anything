#include "OpenvinoEngine.h"

std::vector <float*> OpenvinoEngine::inference(float* data, int n)
{
	std::vector <float*> outBufs;
	auto input_port = compiled_model_->input();
	ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), data);

	infer_request_->set_input_tensor(input_tensor);
	infer_request_->infer();
	for (int i = 0; i < n; i++)
	{
		auto output = infer_request_->get_output_tensor(i); //output(n)
		auto output_shape = output.get_shape();
		//std::cout << "The shape of output:" << output_shape << std::endl;
		float* prob = (float*)output.data<float>();
		outBufs.push_back(prob);
	}

	return outBufs;
}