/*
 * Copyright (C) PerfGao
 */

#include <iostream>
#include "c_ffi.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <unistd.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif
void* lgbm_load(char *model_filename)
{
    try {
        LightGBM::Application *app = new LightGBM::Application(model_filename);
        app->InitPredictOnline();
        return app;
    } catch (const std::exception& ex) {
        std::cerr << "Met Exceptions:" << std::endl;
        std::cerr << ex.what() << std::endl;
        return NULL;
    } catch (const std::string& ex) {
        std::cerr << "Met Exceptions:" << std::endl;
        std::cerr << ex << std::endl;
        return NULL;
    } catch (...) {
        std::cerr << "Unknown Exceptions" << std::endl;
        return NULL;
    }
}

void lgbm_free(LightGBM::Application *app)
{
    try {
        if (app) {
            delete app;
        }
    } catch (...) {
        return;
    }
}

double lgbm_predict(LightGBM::Application *app, double *data, int data_size)
{
    try {
        std::vector<std::pair<int, double>> oneline_features;
        double result = 0;
        int    idx = 0;

        oneline_features.clear();

        for (int i = 0; i < data_size; i++) {
            if (std::fabs(data[i]) > LightGBM::kZeroThreshold || std::isnan(data[i])) {
                oneline_features.emplace_back(idx, data[i]);
            }
            idx++;
        }

        //timeval start_timestamp;
        //gettimeofday(&start_timestamp, NULL);

        app->PredictOnline(oneline_features, &result);

        //timeval end_timestamp;
        //gettimeofday(&end_timestamp, NULL);
        //printf("Time elasped:%lf s\n",(end_timestamp.tv_sec -
        //    start_timestamp.tv_sec) + (end_timestamp.tv_usec -
        //    start_timestamp.tv_usec) / 1000000.0);
        return result;
    } catch (const std::exception& ex) {
        return 0;
    } catch (const std::string& ex) {
        return 0;
    } catch (...) {
        return 0;
    }
}


#ifdef __cplusplus
} //extern "C" {
#endif
