#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <map>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
VideoDataLayer<Dtype>:: ~VideoDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int new_height  = this->layer_param_.video_data_param().new_height();
	const int new_width  = this->layer_param_.video_data_param().new_width();
	const int new_length  = this->layer_param_.video_data_param().new_length();
	const int num_segments = this->layer_param_.video_data_param().num_segments();
	const string& source = this->layer_param_.video_data_param().source();
	const string& source_bbox = this->layer_param_.video_data_param().source_bbox();

	std:: ifstream infile(source.c_str());
	string filename;
	int label;
	int length;

	while (infile >> filename >> length >> label){
		lines_.push_back(std::make_pair(filename,label));
		lines_duration_.push_back(length);

		///////////// for bounding box
		std::vector<string> filename_split;
		boost::split( filename_split, filename, boost::is_any_of( "/" ));
		string video_name = filename_split[filename_split.size()-1];
		string bbox_filename = source_bbox + "/" + video_name + ".txt";
		std::ifstream bbox_infile(bbox_filename.c_str());

		std::vector<std::vector<float> > bboxs;
		int idx_frame;
		string name_class_detector;
		float x_start,y_start,x_end,y_end;
		float prob;
		while(bbox_infile >> idx_frame >> name_class_detector >> x_start >> y_start >> x_end >> y_end >> prob){
			std::vector<float> bbox;
			bbox.push_back(idx_frame);
			bbox.push_back(x_start);
			bbox.push_back(y_start);
			bbox.push_back(x_end);
			bbox.push_back(y_end);
			bboxs.push_back(bbox);
		}
		bbox_pair_.push_back(std::make_pair(filename, bboxs));
		/////////////
	}
	if (this->layer_param_.video_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleVideos();
	}

	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	lines_id_ = 0;

	//check name patter
	if (this->layer_param_.video_data_param().name_pattern() == ""){
		if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB){
			name_pattern_ = "image_%04d.jpg";
		}else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW){
			name_pattern_ = "flow_%c_%04d.jpg";
		}
	}else{
		name_pattern_ = this->layer_param_.video_data_param().name_pattern();
	}

	Datum datum;
	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
	int average_duration = (int) lines_duration_[lines_id_]/num_segments;
	vector<int> offsets;
	for (int i = 0; i < num_segments; ++i){
		caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		int offset = (*frame_rng)() % (average_duration - new_length + 1);
		offsets.push_back(offset+i*average_duration);
	}
	if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW)
		CHECK(ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									 offsets, new_height, new_width, new_length, &datum, name_pattern_.c_str()));
	else
		CHECK(ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									offsets, new_height, new_width, new_length, &datum, true, name_pattern_.c_str()));
	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.video_data_param().batch_size();
	if (crop_size > 0){
		top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
	} else {
		top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
		this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	}
	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
	
	top[1]->Reshape(batch_size, 1, 1, 1);
	this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	this->transformed_data_.Reshape(top_shape);

}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::InternalThreadEntry(){
	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	///////////////// bounding box count
	int count_bbox = 0;
	std::vector<std::vector<float> > batch_bboxs;
	/////////////////

	VideoDataParameter video_data_param = this->layer_param_.video_data_param();
	const int batch_size = video_data_param.batch_size();
	const int new_height = video_data_param.new_height();
	const int new_width = video_data_param.new_width();
	const int new_length = video_data_param.new_length();
	const int num_segments = video_data_param.num_segments();
	const int lines_size = lines_.size();

	std::vector<int> coor_bbox_(batch_size * num_segments * new_length, 1);
	for (int item_id = 0; item_id < batch_size; ++item_id){
		CHECK_GT(lines_size, lines_id_);
		vector<int> offsets;
		int average_duration = (int) lines_duration_[lines_id_] / num_segments;
		for (int i = 0; i < num_segments; ++i){
			if (this->phase_==TRAIN){
				if (average_duration >= new_length){
					caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
					int offset = (*frame_rng)() % (average_duration - new_length + 1);
					offsets.push_back(offset + i * average_duration);
				} else {
					offsets.push_back(1);
				}
			} else{
				if (average_duration >= new_length)
				offsets.push_back(int((average_duration - new_length+1)/2 + i*average_duration));
				else
				offsets.push_back(1);
			}
		}
		if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW){
			if(!ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									   offsets, new_height, new_width, new_length, &datum, name_pattern_.c_str())) {
				continue;
			}
		} else{
			if(!ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									  offsets, new_height, new_width, new_length, &datum, true, name_pattern_.c_str())) {
				continue;
			}
		}
		///////// for bounding box
		int bbox_offset = item_id * (num_segments * new_length);
		bool success = ReadBboxToDatum(offsets, new_length, bbox_pair_[lines_id_].second, &coor_bbox_, bbox_offset);
		if (!success)
			continue;
		std::vector<std::vector<float> > current_bboxs(bbox_pair_[lines_id_].second);
		for(int j = 0; j < current_bboxs.size(); ++j){
			for (int k = 0; k < coor_bbox_.size(); ++k){
				if (coor_bbox_[k] == int(current_bboxs[j][0])){
					++count_bbox;
					std::vector<float> tmp_batch_bbox;
					tmp_batch_bbox.push_back(item_id);
					tmp_batch_bbox.push_back(current_bboxs[j][1]);
					tmp_batch_bbox.push_back(current_bboxs[j][2]);
					tmp_batch_bbox.push_back(current_bboxs[j][3]);
					tmp_batch_bbox.push_back(current_bboxs[j][4]);
					batch_bboxs.push_back(tmp_batch_bbox);
				}
			}
		}
		/////////

		int offset1 = this->prefetch_data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset1);
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
		top_label[item_id] = lines_[lines_id_].second;
		//LOG()

		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_size) {
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if(this->layer_param_.video_data_param().shuffle()){
				ShuffleVideos();
			}
		}
	}
	/////// for bounding box
	prefetch_bbox_.Reshape(count_bbox,5,1,1);
	Dtype* top_rois = prefetch_bbox_.mutable_cpu_data();
	for(int i = 0; i < batch_bboxs.size(); ++i){
		for(int j = 0; j< batch_bboxs[i].size(); ++j){
			top_rois[i * 5 + j] = batch_bboxs[i][j];
		}
	}
	///////
}
template <typename Dtype>
void VideoDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	// First, join the thread
	this->JoinPrefetchThread();
	DLOG(INFO) << "Thread joined";
	// Reshape to loaded data.
	top[0]->ReshapeLike(this->prefetch_data_);
	// Copy the data
	caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
	         top[0]->mutable_cpu_data());
	DLOG(INFO) << "Prefetch copied";
	if (this->output_labels_) {
	// Reshape to loaded labels.
	top[1]->ReshapeLike(this->prefetch_label_);
	// Copy the labels.
	caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
	           top[1]->mutable_cpu_data());
	}

	#ifdef USE_MPI
	//advance (all_rank - (my_rank+1)) mini-batches to be ready for next run
	BaseDataLayer<Dtype>::OffsetCursor(top[0]->num() * (Caffe::MPI_all_rank() - 1));
	#endif
	// Start a new prefetch thread
	DLOG(INFO) << "CreatePrefetchThread";
	this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

template <typename Dtype>
void VideoDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// First, join the thread
	this->JoinPrefetchThread();
	// Reshape to loaded data.
	top[0]->ReshapeLike(this->prefetch_data_);
	// Copy the data
	caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
	top[0]->mutable_gpu_data());
	if (this->output_labels_) {
	// Reshape to loaded labels.
	top[1]->ReshapeLike(this->prefetch_label_);
	// Copy the labels.
	caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
	    top[1]->mutable_gpu_data());
	}
	//////// for bounding box
	top[2]->ReshapeLike(prefetch_bbox_);
	caffe_copy(this->prefetch_bbox_.count(), this->prefetch_bbox_.cpu_data(),
	top[2]->mutable_gpu_data());
	////////
	#ifdef USE_MPI
	//advance (all_rank - (my_rank+1)) mini-batches to be ready for next run
	BaseDataLayer<Dtype>::OffsetCursor(top[0]->num() * (Caffe::MPI_all_rank() - 1));
	#endif
	// Start a new prefetch thread
	this->CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(VideoDataLayer);
}
