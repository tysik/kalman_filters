#pragma once

#include "extended_kalman_filter.h"

namespace kf {

/** \class UnscentedKalmanFilter unscented_kalman_filter.h
 * \brief Class containing functionality of unscented Kalman filter
 *
 * The process is described with equation q(k) = f(q(k-1), u(k-1))
 * The output is described with equation y(k) = h(q(k))
 *
 * Throughout the description "n" denotes dimension of state vector q, "m"
 * denotes dimension of output vector y (measured values), and "l" denotes
 * dimension of input vector u (control signals). Both prediction and correction
 * steps can be invoked separately.
 *
 * The class exploits Armadillo library for matrix operations and can throw any
 * of its exceptions (cf. www.arma.sourceforge.net).
*/
class UnscentedKalmanFilter : public ExtendedKalmanFilter
{
public:

  using KalmanFilter::predictState;
  using KalmanFilter::correctState;

  /** \brief Constructor with given dimensions
   *
   * Initializes all vectors and matrices to have correct dimensions. State
   * matrix as well as all covariance matrices are initialized with ones on
   * diagonal. The rest of data is zero-value initialized.
   *
   * \param dim_in is the dimension of input vector (number of control signals)
   * \param dim_out is the dimension of output vector (number of measurements)
   * \param dim_state is the dimension of state vector
  */
  UnscentedKalmanFilter(size_t dim_in, size_t dim_out, size_t dim_state) :
    ExtendedKalmanFilter(dim_in, dim_out, dim_state),
    k_(2 * dim_state + 1),
    sigma_points_(k_, arma::vec(n_).zeros()),
    pred_sigma_points_(k_, arma::vec(n_).zeros()),
    output_sigma_points_(k_, arma::vec(n_).zeros()),
    mean_weights_(k_, 0.0),
    covariance_weights_(k_, 0.0)
  {}

  /** \brief Constructor with given state-space matrices
   *
   * Initializes all vectors and matrices to have correct dimensions and assigns
   * given state-space matrices. All covariance matrices are initialized with
   * ones on diagonal. The rest of data is zero-value initialized.
   *
   * \param A is the state matrix
   * \param B is the input matrix
   * \param C is the output matrix
  */
  UnscentedKalmanFilter(const arma::mat& A, const arma::mat& B, const arma::mat& C) :
    ExtendedKalmanFilter(A, B, C),
    k_(2 * n_ + 1),
    sigma_points_(k_, arma::vec(n_).zeros()),
    pred_sigma_points_(k_, arma::vec(n_).zeros()),
    output_sigma_points_(k_, arma::vec(n_).zeros()),
    mean_weights_(k_, 0.0),
    covariance_weights_(k_, 0.0)
  {}

  /** \brief Constructor with given state-space vectors
   *
   * Initializes all vectors and matrices to have correct dimensions and assigns
   * given vectors. Both predicted an estimated state vectors are assigned with
   * given state vector. All covariance matrices are initialized with ones on
   * diagonal. The rest of data is zero-value initialized.
   *
   * \param u is the input vector
   * \param y is the output vector
   * \param q is the state vector
  */
  UnscentedKalmanFilter(const arma::vec& u, const arma::vec& y, const arma::vec& q) :
    ExtendedKalmanFilter(u, y, q),
    k_(2 * n_ + 1),
    sigma_points_(k_, arma::vec(n_).zeros()),
    pred_sigma_points_(k_, arma::vec(n_).zeros()),
    output_sigma_points_(k_, arma::vec(n_).zeros()),
    mean_weights_(k_, 0.0),
    covariance_weights_(k_, 0.0)
  {}


  /** \brief Design parameters setter
   *
   * Sets the values of alpha, beta and kappa parameters and calculates the
   * value of parameter lambda. Prepares the weights for mean and covariance
   * propagation via unscented transform.
   * (www.seas.harvard.edu/courses/cs281/papers/unscented.pdf)
   *
   * \param alpha is the new value of alpha parameter
   * \param beta is the new value of beta parameter
   * \param kappa is the new value of kappa parameter
  */
  void setDesignParameters(double alpha, double beta, double kappa) {
    alpha_ = alpha;
    beta_ = beta;
    kappa_ = kappa;
    lambda_ = pow(alpha, 2.0) * (n_ + kappa) - n_;

    // Set the weights for central sigma point
    mean_weights_[0] = lambda_ / (n_ + lambda_);
    covariance_weights_[0] = lambda_ / (n_ + lambda_) + 1.0 - pow(alpha_, 2.0) +
        beta_;

    // Set the weights for other sigma points
    for (size_t i = 1; i < k_; ++i)
      mean_weights_[i] = covariance_weights_[i] = 0.5 / (n_ + lambda_);
  }


  /** \brief Performs the UKF prediction step
    *
    * Calculates the sigma points based on the current covariance matrix and
    * predicts the new state with the use of process function. Propagates the
    * covariance based on unscented transform.
    *
    * Before using this function the design parameters must be set.
    *
    * \sa processFunction_, setDesignParameters()
   */
  virtual void predictState() {
    arma::mat sqrt_P = arma::chol((n_ + lambda_) * P_);

    // 1. Calculate sigma points
    sigma_points_[0] = q_est_;
    for (size_t i = 1; i < n_ + 1; ++i) {
      sigma_points_[i] = q_est_ + sqrt_P.col(i - 1);
      sigma_points_[i + n_] = q_est_ - sqrt_P.col(i - 1);
    }

    // 2. Predict new points based on sigma points
    for (size_t i = 0; i < k_; ++i)
      pred_sigma_points_[i] = processFunction_(sigma_points_[i], u_);

    // 3. Calculate mean of predicted points (weights are normalized)
    q_pred_ = arma::vec(n_).zeros();
    for (size_t i = 0; i < k_; ++i)
      q_pred_ += mean_weights_[i] * pred_sigma_points_[i];

    // 4. Caclulate covariance of predicted points (weights are normalized)
    P_ = Q_;
    for (size_t i = 0; i < k_; ++i)
      P_ += covariance_weights_[i] * (pred_sigma_points_[i] - q_pred_) *
          trans(pred_sigma_points_[i] - q_pred_);
  }

  /** \brief Performs the UKF correction step
   *
   * Propagates the sigma points into output space via output function.
   * Calculates the mean output based on these points and innovation covariance.
   * Calculates the Kalman gain and corrects the prediction with innovation.
   *
   * Before using this function the design parameters must be set.
   *
   * \sa outputFunction_, setDesignParameters()
  */
  virtual void correctState() {
    // 1. Calculate predicted output points based on predicted sigma points
    for (size_t i = 0; i < k_; ++i)
      output_sigma_points_[i] = outputFunction_(pred_sigma_points_[i]);

    // 2. Calculate mean output
    arma::vec y_pred = arma::vec(m_).zeros();
    for (size_t i = 0; i < k_; ++i)
      y_pred += mean_weights_[i] * output_sigma_points_[i];

    // 3. Update innovation covariance matrix
    S_ = R_;
    for (size_t i = 0; i < k_; ++i)
      S_ += covariance_weights_[i] * (output_sigma_points_[i] - y_pred) * trans(output_sigma_points_[i] - y_pred);

    // 4. Calculate cross-covariance
    arma::mat P_qy = arma::mat(n_, m_).zeros();
    for (size_t i = 0; i < k_; ++i)
      P_qy += covariance_weights_[i] * (pred_sigma_points_[i] - q_pred_) * trans(output_sigma_points_[i] - y_pred);

    // 5. Calculate new Kalman gain
    K_ = P_qy * inv(S_);

    // 6. Estimate state and update estimate error covariance matrix
    q_est_ = q_pred_ + K_ * (y_ - y_pred);
    P_ = P_ - K_ * S_ * trans(K_);
  }


protected:

  /** \brief This class cannot be instantiated without providing dimensions. */
  UnscentedKalmanFilter() {}

private:
  /** \brief This base class member function is hidden because it serves no use
   * in this context. */
  void setProcessJacobian() {}

  /** \brief This base class member function is hidden because it serves no use
   * in this context. */
  void setOutputJacobian() {}

  uint k_;          /**< \brief Number of sigma points */

  double alpha_;    /**< \brief Design parameter */
  double beta_;     /**< \brief Design parameter */
  double kappa_;    /**< \brief Design parameter */
  double lambda_;   /**< \brief Automatically calculated parameter */

  std::vector<arma::vec> sigma_points_;         /**< \brief States representing
                                                  the current probability
                                                  distribution */
  std::vector<arma::vec> pred_sigma_points_;    /**< \brief States representing
                                                  the predicted probability
                                                  distribution */
  std::vector<arma::vec> output_sigma_points_;  /**< \brief States representing
                                                  the output probability
                                                  distribution */

  std::vector<double> mean_weights_;        /**< \brief Weights for mean
                                              propagation */
  std::vector<double> covariance_weights_;  /**< \brief Weights for covariance
                                              propagation */
};

} // namespace kf
