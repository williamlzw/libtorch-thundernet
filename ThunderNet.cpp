#include "ThunderNet.h"

torch::Tensor channel_shuffleThunderNet(torch::Tensor x, int64_t groups) {
	auto shape = x.sizes();
	auto batchsize = shape[0];
	auto num_channels = shape[1];
	auto height = shape[2];
	auto width = shape[3];

	auto channels_per_group = num_channels / groups;

	x = x.view({ batchsize, groups, channels_per_group, height, width });
	x = torch::transpose(x, 1, 2).contiguous();
	x = x.view({ batchsize, -1, height, width });

	return x;
}

torch::nn::Sequential conv_bn(int64_t inp, int64_t oup, int64_t stride) {
	auto ret = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(inp, oup, { 3,3 }).stride({ stride,stride }).padding({ 1,1 }).bias(false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup)),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
	);
	return ret;
}

torch::nn::Sequential conv_1x1_bn(int64_t inp, int64_t oup) {
	auto ret = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(inp, oup, { 1,1 }).stride({ 1,1 }).bias(false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup)),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
	);
	return ret;
}

