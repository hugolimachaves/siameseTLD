#include "main.h"

extern "C"{
    void initializer_TLD(char *parameters_path){
        init_TLD(parameters_path);
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
    void TLD_function_2(float *similaridade_positiva_candidates, float *similaridade_negativa_candidates,
                        float *similaridade_positiva_bb_tracker, float *similaridade_negativa_bb_tracker){
        TLD_part_2(similaridade_positiva_candidates, similaridade_negativa_candidates,
                   similaridade_positiva_bb_tracker, similaridade_negativa_bb_tracker);
    }
}
