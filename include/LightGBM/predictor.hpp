#ifndef LIGHTGBM_PREDICTOR_HPP_
#define LIGHTGBM_PREDICTOR_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/boosting.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/dataset.h>
#include <LightGBM/prediction_early_stop.h>

#include <LightGBM/utils/openmp_wrapper.h>

#include <map>
#include <cstring>
#include <cstdio>
#include <vector>
#include <utility>
#include <functional>
#include <string>
#include <memory>
#include <semaphore.h>
namespace LightGBM {

/*!
* \brief Used to predict data with input model
*/
class Predictor {
public:
  /*!
  * \brief Constructor
  * \param boosting Input boosting model
  * \param num_iteration Number of boosting round
  * \param is_raw_score True if need to predict result with raw score
  * \param predict_leaf_index True to output leaf index instead of prediction score
  * \param predict_contrib True to output feature contributions instead of prediction score
  */
  Predictor(Boosting* boosting, int num_iteration,
            bool is_raw_score, bool predict_leaf_index, bool predict_contrib,
            bool early_stop, int early_stop_freq, double early_stop_margin);

  /*!
  * \brief Destructor
  */
  ~Predictor();

  inline const PredictFunction& GetPredictFunction() const {
    return predict_fun_;
  }

  /*!
  * \brief predicting on data, then saving result to disk
  * \param data_filename Filename of data
  * \param result_filename Filename of output result
  */
  void Predict(const char* data_filename, const char* result_filename, bool header);

  void Predict(std::vector<std::pair<int, double>>& oneline_features, double *result);

private:

  void CopyToPredictBuffer(double* pred_buf, const std::vector<std::pair<int, double>>& features);

  void ClearPredictBuffer(double* pred_buf, size_t buf_size, const std::vector<std::pair<int, double>>& features);

  std::unordered_map<int, double> CopyToPredictMap(const std::vector<std::pair<int, double>>& features);

  /*! \brief Boosting model */
  const Boosting* boosting_;
  /*! \brief function for prediction */
  PredictFunction predict_fun_;

  PredictionEarlyStopInstance early_stop_;

  int num_feature_;

  int num_pred_one_row_;

  int num_threads_;

  std::vector<std::vector<double>> predict_buf_;
};


}  // namespace LightGBM

#endif   // LightGBM_PREDICTOR_HPP_
