#include "main.h"

extern "C"{
    void initializer_TLD(char *parameters_path, int* frame,
                         float *array_object_model_positive, int *size_positive,
                         float *array_object_model_negative, int *size_negative,
                         float *array_good_windows, int *size_good_windows,
                         float *array_good_windows_hull, int *size_good_windows_hull){

        init_TLD(parameters_path, frame,
                 array_object_model_positive, size_positive,
                 array_object_model_negative, size_negative,
                 array_good_windows, size_good_windows,
                 array_good_windows_hull, size_good_windows_hull);

    }
    void TLD_function_1(int *frame, float *array_bb_candidates, int *size_candidates,
    					float *array_object_model_positive, int *size_positive,
    					float *array_object_model_negative, int *size_negative,
                        float *bb_tracker, int *size_bb_tracker){

        TLD_part_1(frame, array_bb_candidates, size_candidates,
        		   array_object_model_positive, size_positive,
        		   array_object_model_negative, size_negative,
                   bb_tracker, size_bb_tracker);

    }
    void TLD_function_2(float *similaridade_positiva_candidates, int* size_sim_pos_cand,
                        float *similaridade_negativa_candidates, int* size_sim_neg_cand,
                        float *similaridade_positiva_bb_tracker, int* size_sim_pos_tracker,
                        float *similaridade_negativa_bb_tracker, int* size_sim_neg_tracker,
                        float *array_good_windows,               int *size_good_windows,
                        float *array_good_windows_hull,          int *size_good_windows_hull){

        TLD_part_2(similaridade_positiva_candidates, size_sim_pos_cand,
                   similaridade_negativa_candidates, size_sim_neg_cand,
                   similaridade_positiva_bb_tracker, size_sim_pos_tracker,
                   similaridade_negativa_bb_tracker, size_sim_neg_tracker,
                   array_good_windows,               size_good_windows,
                   array_good_windows_hull,          size_good_windows_hull);

    }
}
