#include "tracking/TLD.hpp"
#include <stdlib.h>

extern "C"{
    void initializer_TLD(char *parameters_path, int* frame,
    					 float *array_object_model_positive, int *size_positive,
                         float *array_object_model_negative, int *size_negative,
    					 float *array_good_windows, int *size_good_windows,
                         float *array_good_windows_hull, int *size_good_windows_hull);

    void TLD_function_1(int *frame, float *array_bb_candidates, int *size_candidates,
    					float *array_object_model_positive, int *size_positive,
    					float *array_object_model_negative, int *size_negative,
    					float *bb_tracker, int *size_bb_tracker);

    void TLD_function_2(float *similaridade_positiva_candidates, float *similaridade_negativa_candidates,
                        float *similaridade_positiva_bb_tracker, float *similaridade_negativa_bb_tracker,
                        float *array_good_windows, int *size_good_windows,
                        float *array_good_windows_hull,  int *size_good_windows_hull);
}
