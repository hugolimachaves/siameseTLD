#ifndef INTEGRATOR_LEARNING_HPP_INCLUDED
#define INTEGRATOR_LEARNING_HPP_INCLUDED

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../common/utils.hpp"
#include "../common/sift_utils.hpp"
#include "Detector.hpp"

using namespace cv;

void initJudge(Mat frame, BoundingBox position, int _valid, int _conf, bool show);
bool IntegratorLearning(Mat frame, BoundingBox t_bb, vector<BoundingBox> detector_positions, vector<double> d_conf,
						bool tracked, bool detected, BoundingBox &output, Mat &object, bool enable_detect,
						float *similaridade_positiva_bb_tracker, int *size_pos_tracker,
						float *similaridade_negativa_bb_tracker, int *size_neg_tracker,
			            float *array_good_windows,               int *size_good_windows,
			            float *array_good_windows_hull,          int *size_good_windows_hull);

#endif // INTEGRATOR_LEARNING_HPP_INCLUDED

