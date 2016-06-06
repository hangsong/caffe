#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using std::max;
using std::min;

namespace caffe {
template <typename Dtype>
__global__ void ATTENTIONPoolForward(const int nthreads, const Dtype* bottom_data,
    const int channels, const int height,
    const int width, Dtype* top_data, const Dtype* rois_data, const int count_rois_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int idx_batch = index / (channels * height * width);
    int idx_height = (index % (height * width))/ width;
    int idx_width = index % width;
    top_data[index] = bottom_data[index] * rois_data[idx_batch * height * width + idx_height * width + idx_width];
  }
}
template <typename Dtype>
void AttentionPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int top_count = top[0]->count();
  Dtype spatial_scale_ = 1.0/32.0;

  caffe_gpu_set(top_count, Dtype(0), top_data);
  int num_rois = bottom[1]->num();
  // parameter init
  const int count_rois_data = num_ * height_ * width_;
  Dtype init_attention_weight = 1.0 / (height_ * width_);
  std::vector<Dtype> attention_weight(num_ * height_ * width_, 1.0);
  std::vector<int> histogram(num_ * height_ * width_, 0);
  std::vector<int> total_count_num(num_, 0);
  // ori_roi scaled to feature map size
  for (int n = 0; n < num_rois; ++n) {
    int idx_batch = bottom_rois[roi_channel_ * n];
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
    total_count_num[idx_batch] += (roi_height * roi_width);
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
  Dtype* rois_data_ = tmp_rois_data_.mutable_cpu_data();
  for(int i = 0; i < count_rois_data; ++i){
    rois_data_[i] = attention_weight[i];
  }
  const Dtype* rois_data = tmp_rois_data_.gpu_data();
  ATTENTIONPoolForward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
      top_count, bottom_data, channels_, height_, width_, top_data, rois_data, count_rois_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ATTENTIONPoolBackward(const int nthreads, Dtype* bottom_diff,
    const int channels, const int height,
    const int width, const Dtype* top_diff, const Dtype* rois_data, const int count_rois_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int idx_batch = index / (channels * height * width);
    int idx_height = (index % (height * width))/ width;
    int idx_width = index % width;
    bottom_diff[index] = top_diff[index] * rois_data[idx_batch * height * width + idx_height * width + idx_width];
  }
}
template <typename Dtype>
void AttentionPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down ,const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  int bottom_count = bottom[0]->count();
  caffe_gpu_set(bottom_count, Dtype(0.), bottom_diff);

  Dtype spatial_scale_ = 1.0/32.0;
  int num_rois = bottom[1]->num();
  // parameter init
  const int count_rois_data = num_ * height_ * width_;
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
    int roi_start_x = min(max(roi_start_x_, 0), height_ - 1);
    int roi_start_y = min(max(roi_start_y_, 0), width_ - 1);
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
  Dtype* rois_data_ = tmp_rois_data_.mutable_cpu_data();
  for(int i = 0; i < count_rois_data; ++i){
    rois_data_[i] = attention_weight[i];
  }
  const Dtype* rois_data = tmp_rois_data_.gpu_data();
  ATTENTIONPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
      bottom_count, bottom_diff, channels_, height_, width_, top_diff, rois_data, count_rois_data);
  CUDA_POST_KERNEL_CHECK;
}
 
INSTANTIATE_LAYER_GPU_FUNCS(AttentionPoolingLayer);

}
