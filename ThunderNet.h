#pragma once
#include <torch/torch.h>

torch::Tensor channel_shuffleThunderNet(torch::Tensor x, int64_t groups);
torch::nn::Sequential conv_bn(int64_t inp, int64_t oup, int64_t stride);
torch::nn::Sequential conv_1x1_bn(int64_t inp, int64_t oup);


struct InvertedResidualImpl : torch::nn::Module {
	InvertedResidualImpl(int64_t inp, int64_t oup, int64_t stride, int64_t benchmodel) {
		_benchmodel = benchmodel;
		_stride = stride;
		oup_inc = oup / 2;
		if (benchmodel == 1) {//basic unit
			branch2 = torch::nn::Sequential(
				//pw
				torch::nn::Conv2d(torch::nn::Conv2dOptions(oup_inc, oup_inc, { 1,1 }).stride({ 1,1 }).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup_inc)),
				torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
				//dw
				torch::nn::Conv2d(torch::nn::Conv2dOptions(oup_inc, oup_inc, { 5,5 }).stride({ stride,stride }).padding({ 2,2 }).groups(oup_inc).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup_inc)),
				//pw-linear
				torch::nn::Conv2d(torch::nn::Conv2dOptions(oup_inc, oup_inc, { 1,1 }).stride({ 1,1 }).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup_inc)),
				torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
				);
		}
		else {//down sample (2x)
			branch1 = torch::nn::Sequential(
				
				//dw
				torch::nn::Conv2d(torch::nn::Conv2dOptions(inp, inp, { 5,5 }).stride({ stride,stride }).padding({ 2,2 }).groups(inp).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(inp)),
				//pw-linear
				torch::nn::Conv2d(torch::nn::Conv2dOptions(inp, oup_inc, { 1,1 }).stride({ 1,1 }).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup_inc)),
				torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
			);
			branch2 = torch::nn::Sequential(
				//pw
				torch::nn::Conv2d(torch::nn::Conv2dOptions(inp, oup_inc, { 1,1 }).stride({ 1,1 }).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup_inc)),
				torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
				//dw
				torch::nn::Conv2d(torch::nn::Conv2dOptions(oup_inc, oup_inc, { 5,5 }).stride({ stride,stride }).padding({ 2,2 }).groups(oup_inc).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup_inc)),
				//pw-linear
				torch::nn::Conv2d(torch::nn::Conv2dOptions(oup_inc, oup_inc, { 1,1 }).stride({ 1,1 }).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup_inc)),
				torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
			);
		}
		
		register_module("branch2", branch2);
	
		if (!branch1.is_empty()) {
			register_module("branch1", branch1);
		}
		
		
	}
	torch::Tensor forward(torch::Tensor x) {
		torch::Tensor out;
		if (_benchmodel == 1) {
			auto chunks = x.chunk(2, 1);
			out = torch::cat({ chunks[0], branch2->forward(chunks[1]) }, 1);
		}
		else if (_benchmodel == 2) {
			out = torch::cat({ branch1->forward(x), branch2->forward(x) }, 1);
		}
		out = channel_shuffleThunderNet(out, 2);
		return out;
	}

	int64_t _benchmodel;
	int64_t _stride;
	int64_t oup_inc;
	torch::nn::Sequential branch1{ nullptr };
	torch::nn::Sequential branch2{ nullptr };
};

TORCH_MODULE(InvertedResidual);

struct SNet49Impl : torch::nn::Module {
	SNet49Impl(int64_t n_class = 1000, int64_t input_size = 224) {
		assert(input_size % 32 == 0);
		std::vector<int64_t> stage_repeats = { 4,8,4 };
		 stage_out_channels = { -1, 24, 60, 120, 240, 512 };
		//building first layer
		int64_t input_channel = stage_out_channels[1];
		conv1 = conv_bn(3, input_channel, 2);
		maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ 3,3 }).stride({ 2,2 }).padding({1,1}));
		//stage2
		int64_t numrepeat2 = stage_repeats[0];
		int64_t output_channel2= stage_out_channels[2];
		
		for (int64_t i = 0; i < numrepeat2; i++) {
			if (i == 0) {
				features1->push_back(InvertedResidual(InvertedResidualImpl(input_channel, output_channel2, 2, 2)));
				
			}
			else {
				features1->push_back(InvertedResidual(InvertedResidualImpl(input_channel, output_channel2, 1, 1)));
			}
			
			input_channel = output_channel2;
		}
		//stage3
		
		int64_t numrepeat3 = stage_repeats[1];
		int64_t output_channel3 = stage_out_channels[3];
		for (int64_t i = 0; i < numrepeat3; i++) {
			if (i == 0) {
				features2->push_back(InvertedResidual(InvertedResidualImpl(input_channel, output_channel3, 2, 2)));

			}
			else {
				features2->push_back(InvertedResidual(InvertedResidualImpl(input_channel, output_channel3, 1, 1)));
			}
			input_channel = output_channel3;
		}
		//stage4
		int64_t numrepeat4 = stage_repeats[2];
		int64_t output_channel4 = stage_out_channels[4];
		
		for (int64_t i = 0; i < numrepeat4; i++) {
			if (i == 0) {
				features3->push_back(InvertedResidual(InvertedResidualImpl(input_channel, output_channel4, 2, 2)));

			}
			else {
				features3->push_back(InvertedResidual(InvertedResidualImpl(input_channel, output_channel4, 1, 1)));
			}
			input_channel = output_channel4;
		}
		//building last several layers
		conv5 = conv_1x1_bn(input_channel, stage_out_channels[5]);
		globalpool = torch::nn::Sequential(
			torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ input_size / 32,input_size / 32 }))
		);
		//building classifier
		classifier = torch::nn::Sequential(
			torch::nn::Linear(torch::nn::LinearOptions(stage_out_channels[5], n_class))
		);
		register_module("conv1", conv1);
		register_module("conv5", conv5);
		register_module("maxpool", maxpool);
		register_module("features1", features1);
		register_module("features2", features2);
		register_module("features3", features3);
		register_module("globalpool", globalpool);
		register_module("classifier", classifier);
	}
	torch::Tensor forward(torch::Tensor x) {
		x = conv1->forward(x);
		x = maxpool->forward(x);
		x = features1->forward(x);
		x = features2->forward(x);
		//auto out_c4 = x;
		x = features3->forward(x);
		x = conv5->forward(x);
		//auto out_c5 = x;
		x = globalpool->forward(x);
		x = x.view({ -1,stage_out_channels[5] });
		x = classifier->forward(x);
		return x;
	}

	torch::nn::Sequential conv1{ nullptr };
	torch::nn::Sequential conv5{ nullptr };
	torch::nn::MaxPool2d maxpool{ nullptr };
	torch::nn::Sequential features1;
	torch::nn::Sequential features2;
	torch::nn::Sequential features3;
	torch::nn::Sequential globalpool{ nullptr };
	torch::nn::Sequential classifier{ nullptr };
	std::vector<int64_t> stage_out_channels;
};

TORCH_MODULE(SNet49);

struct SNet146Impl : torch::nn::Module {
	SNet146Impl(int64_t n_class = 1000, int64_t input_size = 224) {
		assert(input_size % 32 == 0);
		std::vector<int64_t> stage_repeats = { 4,8,4 };
		stage_out_channels = { -1, 24, 132, 264, 528, 1024 };
		//building first layer
		int64_t input_channel = stage_out_channels[1];
		conv1 = conv_bn(3, input_channel, 2);
		maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ 3,3 }).stride({ 2,2 }).padding({ 1,1 }));
		//building inverted residual blocks
		for (int64_t idxstage = 0; idxstage < 3; idxstage++) {
			int64_t numrepeat = stage_repeats[idxstage];
			int64_t output_channel = stage_out_channels[idxstage + 2];
			for (int64_t i = 0; i < numrepeat; i++) {
				if (i == 0) {
					features->push_back(InvertedResidual(InvertedResidualImpl(input_channel, output_channel, 2, 2)));

				}
				else {
					features->push_back(InvertedResidual(InvertedResidualImpl(input_channel, output_channel, 1, 1)));
				}
				input_channel = output_channel;
			}
		}
		//building last several layers
		globalpool = torch::nn::Sequential(
			torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ input_size / 32,input_size / 32 }))
		);
		//building classifier
		classifier = torch::nn::Sequential(
			torch::nn::Linear(torch::nn::LinearOptions(stage_out_channels[5], n_class))
		);
		register_module("conv1", conv1);
		register_module("maxpool", maxpool);
		register_module("features", features);
		register_module("globalpool", globalpool);
		register_module("classifier", classifier);
	}
	torch::Tensor forward(torch::Tensor x) {
		x = conv1->forward(x);
		x = maxpool->forward(x);
		x = features->forward(x);
		x = globalpool->forward(x);
		x = x.view({ -1,stage_out_channels[5] });
		x = classifier->forward(x);
		return x;
	}
	torch::nn::Sequential conv1{ nullptr };
	torch::nn::MaxPool2d maxpool{ nullptr };
	torch::nn::Sequential features;
	torch::nn::Sequential globalpool{ nullptr };
	torch::nn::Sequential classifier{nullptr};
	std::vector<int64_t> stage_out_channels;
};

TORCH_MODULE(SNet146);

struct SNet535Impl : torch::nn::Module {
	SNet535Impl(int64_t n_class = 1000, int64_t input_size = 224) {
		assert(input_size % 32 == 0);
		std::vector<int64_t> stage_repeats = { 4,8,4 };
		stage_out_channels = { -1, 48, 248, 496, 992, 1024 };
		//building first layer
		int64_t input_channel = stage_out_channels[1];
		conv1 = conv_bn(3, input_channel, 2);
		maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ 3,3 }).stride({ 2,2 }).padding({ 1,1 }));
		//building inverted residual blocks
		for (int64_t idxstage = 0; idxstage < 3; idxstage++) {
			int64_t numrepeat = stage_repeats[idxstage];
			int64_t output_channel = stage_out_channels[idxstage + 2];
			for (int64_t i = 0; i < numrepeat; i++) {
				if (i == 0) {
					features->push_back(InvertedResidual(InvertedResidualImpl(input_channel, output_channel, 2, 2)));

				}
				else {
					features->push_back(InvertedResidual(InvertedResidualImpl(input_channel, output_channel, 1, 1)));
				}
				input_channel = output_channel;
			}
		}
		//building last several layers
		globalpool = torch::nn::Sequential(
			torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ input_size / 32,input_size / 32 }))
		);
		//building classifier
		classifier = torch::nn::Sequential(
			torch::nn::Linear(torch::nn::LinearOptions(stage_out_channels[5], n_class))
		);
		register_module("conv1", conv1);
		register_module("maxpool", maxpool);
		register_module("features", features);
		register_module("globalpool", globalpool);
		register_module("classifier", classifier);
	}
	torch::Tensor forward(torch::Tensor x) {
		x = conv1->forward(x);
		x = maxpool->forward(x);
		x = features->forward(x);
		x = globalpool->forward(x);
		x = x.view({ -1,stage_out_channels[5] });
		x = classifier->forward(x);
		return x;
	}
	torch::nn::Sequential conv1{ nullptr };
	torch::nn::MaxPool2d maxpool{ nullptr };
	torch::nn::Sequential features;
	torch::nn::Sequential globalpool{ nullptr };
	torch::nn::Sequential classifier{ nullptr };
	std::vector<int64_t> stage_out_channels;
};

TORCH_MODULE(SNet535);


template <typename DataLoader>
void trainThunderNet(int32_t epoch, size_t batch_size, SNet49 model, torch::Device device, DataLoader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size)
{
	model->train();
	size_t batch_idx = 0;
	for (auto& batch : data_loader) {
		auto data = batch.data.to(device);
		auto targets = batch.target.to(device);
		targets = targets.reshape(batch_size);
		optimizer.zero_grad();
		auto output = model->forward(data);
		output = torch::log_softmax(output, 1);
		auto loss = torch::nll_loss(output, targets);
		loss.backward();
		optimizer.step();
		batch_idx++;
		if (batch_idx % 10 == 0) {
			std::printf(
				"\r epoch: %ld [%5ld/%5ld] Loss: %.6f",
				epoch,
				batch_idx * batch.data.size(0),
				dataset_size,
				loss.item<float>());
		}
		else {
			std::printf(
				"\r epoch: %ld [%5ld/%5ld] Loss: %.6f",
				epoch,
				batch_idx * batch.data.size(0),
				dataset_size,
				loss.item<float>());
		}
	}
}
