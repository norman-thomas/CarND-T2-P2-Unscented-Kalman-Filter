#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd rmse = VectorXd::Zero(4);
  
  if (estimations.size() != ground_truth.size()) {
    throw "size_mismatch";
  }

  if (estimations.size() == 0) {
    return rmse;
  }
  
  for (unsigned int i = 0; i < estimations.size(); i++) {
    VectorXd diff = estimations[i] - ground_truth[i];
    diff = diff.array().square();
    rmse += diff;
  }

  rmse /= estimations.size();

  return rmse.array().sqrt();
}