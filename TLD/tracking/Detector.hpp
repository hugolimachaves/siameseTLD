#ifndef DETECTOR_HPP_INCLUDED
#define DETECTOR_HPP_INCLUDED

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "../common/utils.hpp"

#define NUM_FERNS		   10	  //Numero de ferns. NUM_FERNS*NUM_FEAT_FERN < DEFAULT_PATCH_SIZE*DEFAULT_PATCH_SIZE*(DEFAULT_PATCH_SIZE-1)
#define _DEBUG_PERF 		0		//Imprimir tempo de cada processo da detecção
#define _DEBUG_WORKSPACE 	0		//Mostrar features, windows, respostas...

using namespace cv;

class ModelSample{
	public:
		Mat image;
		Mat ens_img;
		Mat nn_img;

		int code[NUM_FERNS];

		ModelSample();
		~ModelSample();
		double similarity(Mat pattern2);
};

class Candidate{
	public:
		Mat image;
		Mat ens_img;
		Mat nn_img;

		int code[NUM_FERNS];

		float variance;
		float average_vote;
		double r_sim;
		double c_sim;
		int scanning_windows_index;

		Candidate();
		~Candidate();
};

void Train(Mat frame, BoundingBox &position, bool show,
		   float *array_object_model_positive, int *size_positive,
		   float *array_object_model_negative, int *size_negative,
           float *array_good_windows, int *size_good_windows,
           float *array_good_windows_hull, int *size_good_windows_hull);

bool Retrain(Mat frame, BoundingBox &position, float *similaridade_positiva_bb_tracker,
			 float *similaridade_negativa_bb_tracker, bool show,
             float *array_good_windows, int *size_good_windows,
             float *array_good_windows_hull, int *size_good_windows_hull);

bool Detect_part_1(Mat frame, int frame_number, /*string saidaTemplates,*/
				   float *array_bb_candidates, int *size_candidates,
				   float *array_object_model_positive, int *size_positive,
				   float *array_object_model_negative, int *size_negative);

bool Detect_part_2(Mat frame, vector<BoundingBox> &detector_positions, vector<double> &d_conf, int frame_number,
				   float *similaridade_positiva_candidates, float *similaridade_negativa_candidates);

void DetClear();

void unnorm_object_model_clear();

void normalize(Mat img, Mat blur_img, BoundingBox bb, float shift_x, float shift_y, Mat &sample, Mat &ens_img, Mat &nn_img);

double conservativeSimilarity(Mat pattern, float *similaridade_positiva_bb_tracker, float *similaridade_negativa_bb_tracker);

#endif // DETECTOR_HPP_INCLUDED
