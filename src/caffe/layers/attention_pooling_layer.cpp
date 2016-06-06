#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
namespace caffe {

using std::max;
using std::min;
using std::floor;
using std::ceil;

template <typename Dtype>
void AttentionPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  AttentionPoolingParameter attention_pooling_param = this->layer_param_.attention_pooling_param();
  scale_ = attention_pooling_param.scale();
  LOG(INFO) << "Attention pooling scale is: " << scale_;
}

template <typename Dtype>
void AttentionPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		roi_channel_ = bottom[1]->channels();
		num_ = bottom[0]->num();
		channels_ = bottom[0]->channels();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
    tmp_rois_data_.Reshape(num_ * height_ * width_, 1, 1, 1);

}

template <typename Dtype>
void AttentionPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// fixed paramater
	Dtype spatial_scale_ = 1.0/32.0;
	// foundation parameter
	int top_count = top[0]->count();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_rois = bottom[1]->cpu_data();
	caffe_set(top_count, Dtype(0), top_data);
	int num_rois = bottom[1]->num();
	// parameter init
	Dtype init_attention_weight = 1.0 / (height_ * width_);
	std::vector<int> histogram(num_ * height_ * width_, 0);
	std::vector<Dtype> attention_weight(num_ * height_ * width_, 1.0);
	std::vector<int> total_count_num(num_, 0);
	// ori_roi scaled to feature map size
	for (int n = 0; n < num_rois; ++n) {
		int idx_batch = bottom_rois[roi_channel_*n];
		int roi_start_x_ = static_cast<int>(floor(bottom_rois[1 + n * roi_channel_] * spatial_scale_));
		int roi_start_y_ = static_cast<int>(floor(bottom_rois[2 + n * roi_channel_] * spatial_scale_));
		int roi_end_x_ = static_cast<int>(ceil(bottom_rois[3 + n * roi_channel_] * spatial_scale_));
		int roi_end_y_ = static_cast<int>(ceil(bottom_rois[4 + n * roi_channel_] * spatial_scale_));
		int roi_start_x = min(max(roi_start_x_, 0), width_ - 1);
		int roi_start_y = min(max(roi_start_y_, 0), height_ - 1);
		int roi_end_x = min(max(roi_end_x_, 0), width_ - 1);
		int roi_end_y = min(max(roi_end_y_, 0), height_ - 1);

		int roi_height = roi_end_y - roi_start_y + 1;
		int roi_width = roi_end_x - roi_start_x + 1;
		total_count_num[idx_batch] += roi_height * roi_width;

		//histogram calculator
		for (int coor_index_h = roi_start_y; coor_index_h <= roi_end_y; coor_index_h++){
			for (int coor_index_w = roi_start_x; coor_index_w <= roi_end_x; coor_index_w++){
				int attention_count_index = idx_batch * height_ * width_ + coor_index_h * width_ + coor_index_w;
				histogram[attention_count_index] += 1;
			}
		}
	}
	// attention weight calculator
	for (int idx_batch = 0; idx_batch < num_; idx_batch++){
		for (int i = 0; i < height_; i++){
			for (int j = 0; j < width_; j++){
				int idx_attention = idx_batch * height_ * width_ + i * width_ + j;
				attention_weight[idx_attention] = 
					(init_attention_weight + (scale_ * (histogram[idx_attention] / total_count_num[idx_batch]))) / (1.0 + scale_);
			}
		}
	}
	// top_data calculator
	for (int idx_batch = 0; idx_batch < num_; idx_batch++){
		for (int c = 0; c < channels_; c++){
			for (int i = 0; i < height_; i++){
				for (int j = 0; j < width_; j++){
					int data_index = idx_batch * channels_ * height_ * width_ + c * height_ * width_ + i * width_ + j;
					top_data[data_index] = bottom_data[data_index] * attention_weight[idx_batch * height_* width_ + i * width_ + j];
				}
			}
		}
	}
}

template <typename Dtype>
void AttentionPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// foundation parameter
	const Dtype* top_diff = top[0]->cpu_diff();
  	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  	// fixed paramater
	Dtype spatial_scale_ = 1.0/32.0;
	// foundation parameter
	const Dtype* bottom_rois = bottom[1]->cpu_data();
	int num_rois = bottom[1]->num();

	// parameter init
	Dtype init_attention_weight = 1.0 / (height_ * width_);
	std::vector<int> histogram(num_ * height_ * width_, 0);
	std::vector<Dtype> attention_weight(num_ * height_ * width_, 1.0);
	std::vector<int> total_count_num(num_, 0);
	// ori_roi scaled to feature map size
	for (int n = 0; n < num_rois; ++n) {
		int idx_batch = bottom_rois[roi_channel_*n];
		int roi_start_x_ = static_cast<int>(floor(bottom_rois[1 + n * roi_channel_] * spatial_scale_));
		int roi_start_y_ = static_cast<int>(floor(bottom_rois[2 + n * roi_channel_] * spatial_scale_));
		int roi_end_x_ = static_cast<int>(ceil(bottom_rois[3 + n * roi_channel_] * spatial_scale_));
		int roi_end_y_ = static_cast<int>(ceil(bottom_rois[4 + n * roi_channel_] * spatial_scale_));
		int roi_start_x = min(max(roi_start_x_, 0), width_ - 1);
		int roi_start_y = min(max(roi_start_y_, 0), height_ - 1);
		int roi_end_x = min(max(roi_end_x_, 0), width_ - 1);
		int roi_end_y = min(max(roi_end_y_, 0), height_ - 1);

		int roi_height = roi_end_y - roi_start_y + 1;
		int roi_width = roi_end_x - roi_start_x + 1;
	
		// total_count_num count
		total_count_num[idx_batch] += roi_height * roi_width;

		//histogram calculator
		for (int coor_index_h = roi_start_y; coor_index_h <= roi_end_y; coor_index_h++){
			for (int coor_index_w = roi_start_x; coor_index_w <= roi_end_x; coor_index_w++){
				int attention_count_index = idx_batch * height_ * width_ + coor_index_h * width_ + coor_index_w;
				histogram[attention_count_index] += 1;
			}
		}
	}
	// attention weight calculator
	for (int idx_batch = 0; idx_batch < num_; idx_batch++){
		// normalizaiton denominator
		for (int i = 0; i < height_; i++){
			for (int j = 0; j < width_; j++){
				int idx_attention = idx_batch * height_ * width_ + i * width_ + j;
				attention_weight[idx_attention] =
					(init_attention_weight + (scale_ * (histogram[idx_attention] / total_count_num[idx_batch]))) / (1.0 + scale_);
			}
		}
	}
	// main loop
	for (int idx_batch = 0; idx_batch < num_; ++idx_batch) {
		for (int c = 0; c < channels_; ++c) {
			for (int i = 0; i < height_; ++i) {
				for (int j = 0; j < width_; ++j) {
					int diff_index = idx_batch * channels_ * height_ * width_ + c * height_ * width_ + i * width_ + j;
					bottom_diff[diff_index] = top_diff[diff_index] * attention_weight[idx_batch * height_ * width_ + i * width_ + j];
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(AttentionPoolingLayer);
#endif

INSTANTIATE_CLASS(AttentionPoolingLayer);
REGISTER_LAYER_CLASS(AttentionPooling);

}  // namespace caffe
