/*
 * Copyright (C) PerfGao
 */

#ifndef __C_FFI_H__
#define __C_FFI_H__

#include <LightGBM/application.h>
#include <LightGBM/meta.h>

#ifdef __cplusplus
extern "C" {
#endif
void *lgbm_load(char *model_filename);
void lgbm_free(LightGBM::Application *app);

double lgbm_predict(LightGBM::Application *app, double *data, int data_size);

void lgbm_multi_predict(LightGBM::Application *app, char **data, int data_size);
#ifdef __cplusplus
}
#endif


#endif
