#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define EPSILON 0.001

/**
* Initializes Unscented Kalman filter
*/
UKF::UKF() {
  
  n_x_ = 5;
  n_aug_ = 7;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
  
  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  
  // initial state vector
  x_ = VectorXd(n_x_);
  
  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;
  
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
  
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;
  
  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
  
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;
  
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
  TODO:
  
  Complete the initialization. See ukf.h for other member properties.
  
  Hint: one or more values initialized above might be wildly off...
  */
  
  is_initialized_ = false;
  time_us_ = 0ll;
  
  n_z_radar_ = 3;
  n_z_lidar_ = 2;
  lambda_ = 3 - n_x_;

  nis_lidar_ = 0.0;
  nis_radar_ = 0.0;
  
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd(2 * n_aug_ + 1);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_[0] = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    double weight = 0.5/(n_aug_ + lambda_);
    weights_[i] = weight;
  }
  
  MatrixXd R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0, std_radrd_*std_radrd_;
}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:
  
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
      InitializeRadar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
      InitializeLidar(meas_package);
    }

    // adjusting initial values close to zero:
    if ((fabs(x_[0]) < EPSILON) && (fabs(x_[1]) < EPSILON)) {
      x_[0] = EPSILON;
      x_[1] = EPSILON;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
}

void UKF::InitializeRadar(const MeasurementPackage& meas_package) {
  const double rho = meas_package.raw_measurements_[0];
  const double phi = meas_package.raw_measurements_[1];
  const double rho_dot = meas_package.raw_measurements_[2];

  const double px = rho*cos(phi);
  const double py = rho*sin(phi);

  const double vx = rho_dot*cos(phi);
  const double vy = rho_dot*sin(phi);
  const double v = sqrt(vx*vx + vy*vy);

  x_ << px, py, v, phi, 0.0;
}

void UKF::InitializeLidar(const MeasurementPackage& meas_package) {
  const double px = meas_package.raw_measurements_[0];
  const double py = meas_package.raw_measurements_[1];
  const double phi = atan2(py, px);

  x_ << px, py, 0.0, phi, 0.0;
}

/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t) {
  /**
  TODO:
  
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  MatrixXd Xsig = GenerateSigmaPoints();
  Xsig_pred_ = PredictSigmaPoints(Xsig, delta_t);
  PredictMeanAndCovariance();
}

/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:
  
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  
  You'll also need to calculate the lidar NIS.
  */

  MatrixXd Zsig;
  VectorXd z_pred;
  MatrixXd S;
  PredictLidarMeasurement(Zsig, z_pred, S);

  const VectorXd z_diff = UpdateState(meas_package, Zsig, z_pred, S, n_z_lidar_);
  nis_lidar_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:
  
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  
  You'll also need to calculate the radar NIS.
  */

  MatrixXd Zsig;
  VectorXd z_pred;
  MatrixXd S;
  PredictRadarMeasurement(Zsig, z_pred, S);

  const VectorXd z_diff = UpdateState(meas_package, Zsig, z_pred, S, n_z_radar_);
  nis_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}


MatrixXd UKF::GenerateSigmaPoints() const {
  VectorXd X_aug = VectorXd::Zero(n_aug_);
  X_aug.head(n_x_) = x_;
  
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_aug_ - 2, n_aug_ - 2) = std_a_ * std_a_;
  P_aug(n_aug_ - 1, n_aug_ - 1) = std_yawdd_ * std_yawdd_;
  
  const MatrixXd A = P_aug.llt().matrixL();
  MatrixXd Xsig = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig.col(0)  = X_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig.col(i + 1) = X_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig.col(i + 1 + n_aug_) = X_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }
  
  return Xsig;
}

MatrixXd UKF::PredictSigmaPoints(const MatrixXd& Xsig, const double delta_t) const {
  MatrixXd Xsig_pred = MatrixXd(n_x_ , 2 * n_aug_ + 1);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    const double p_x = Xsig(0, i);
    const double p_y = Xsig(1, i);
    const double v = Xsig(2, i);
    const double yaw = Xsig(3, i);
    const double yawd = Xsig(4, i);
    const double nu_a = Xsig(5, i);
    const double nu_yawdd = Xsig(6, i);
    
    // predicted state values
    double px_p, py_p;
    
    // avoid division by zero
    if (fabs(yawd) > EPSILON) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }
    
    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;
    
    // add noise
    px_p += 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p += 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p += nu_a*delta_t;
    
    yaw_p += 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p += nu_yawdd*delta_t;
    
    // write predicted sigma point into right column
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }
  return Xsig_pred;
}

void UKF::PredictMeanAndCovariance() {
  x_ = VectorXd::Zero(n_x_);
  // predicted state mean
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ += weights_[i] * Xsig_pred_.col(i);
  }
  
  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff[3] = NormalizeAngle(x_diff[3]);
    
    P_ += weights_[i] * x_diff * x_diff.transpose() ;
  }
}

double UKF::NormalizeAngle(double angle) const {
  while (angle > M_PI) {
    angle -= 2. * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2. * M_PI;
  }
  return angle;
}

void UKF::PredictRadarMeasurement(MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S) {
  const unsigned int n_z = n_z_radar_;
  
  Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    const double p_x = Xsig_pred_(0, i);
    const double p_y = Xsig_pred_(1, i);
    const double v  = Xsig_pred_(2, i);
    const double yaw = Xsig_pred_(3, i);
    
    const double v1 = cos(yaw)*v;
    const double v2 = sin(yaw)*v;
    
    // measurement model
    Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1, i) = atan2(p_y, p_x);                                //phi
    Zsig(2, i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }
  
  // mean predicted measurement
  z_pred = VectorXd::Zero(n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_[i] * Zsig.col(i);
  }
  
  // measurement covariance matrix S
  S = MatrixXd::Zero(n_z, n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // angle normalization
    z_diff[1] = NormalizeAngle(z_diff[1]);
    
    S += weights_[i] * z_diff * z_diff.transpose();
  }
  
  //add measurement noise covariance matrix
  S += R_radar_;
}

void UKF::PredictLidarMeasurement(MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S) {
  // set measurement dimension
  int n_z = n_z_lidar_;
  
  // matrix for sigma points in measurement space
  Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  
  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    
    if (fabs(p_x) < EPSILON || fabs(p_y) < EPSILON) {
      p_x = EPSILON;
      p_y = EPSILON;
    }
    
    // measurement model
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
  }
  
  // mean predicted measurement
  z_pred = VectorXd::Zero(n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_[i] * Zsig.col(i);
  }
  
  // measurement covariance matrix S
  S = MatrixXd::Zero(n_z, n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    S += weights_[i] * z_diff * z_diff.transpose();
  }
  
  // add measurement noise covariance matrix
  // some people call this Tc
  S += R_lidar_;
}

VectorXd UKF::UpdateState(const MeasurementPackage& meas_package, const MatrixXd& Zsig, const VectorXd& z_pred, const MatrixXd& S, const unsigned int n_z) {
  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff[1] = NormalizeAngle(z_diff[1]);
    

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff[3] = NormalizeAngle(x_diff[3]);
    
    Tc += weights_[i] * x_diff * z_diff.transpose();
  }
  
  const MatrixXd K = Tc * S.inverse();
  const VectorXd z = meas_package.raw_measurements_;
  
  VectorXd z_diff = z - z_pred;
  
  z_diff[1] = NormalizeAngle(z_diff[1]);
  
  //update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  return z_diff;
}
