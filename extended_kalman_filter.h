#pragma once

#include "kalman_filter.h"

namespace kf {

/** \class ExtendedKalmanFilter extended_kalman_filter.h
 * \brief Class containing functionality of extended Kalman filter.
 *
 * Class containing functionality of extended Kalman filter. Both prediction and
 * correction steps can be invoked separately.
*/
class ExtendedKalmanFilter : public KalmanFilter
{
public:
  /** \brief Constructor with given dimensions
   *
   * Constructor with given dimensions. Initializes all vectors and matrices to
   * have correct dimensions. State matrix as well as all covariance matrices
   * are initialized with ones on diagonal. The rest of data is zero-value
   * initialized.
   *
   * \param dim_in the dimension of input vector (number of control signals)
   * \param dim_out the dimension of output vector (number of measurements)
   * \param dim_state the dimension of state vector
  */
  ExtendedKalmanFilter(uint dim_in, uint dim_out, uint dim_state) :
    KalmanFilter(dim_in, dim_out, dim_state),
    W_(arma::eye(n_, n_)),
    V_(arma::eye(m_, m_))
  {}

  /** \brief Constructor with given state-space matrices
   *
   * Constructor with given state-space matrices. Initializes all vectors and
   * matrices to have correct dimensions and assigns given state-space matrices.
   * All covariance matrices are initialized with ones on diagonal. The rest
   * of data is zero-value initialized.
   *
   * \param A the state matrix
   * \param B the input matrix
   * \param C the output matrix
  */
  ExtendedKalmanFilter(const arma::mat& A, const arma::mat& B, const arma::mat& C) :
    KalmanFilter(A, B, C),
    W_(arma::eye(n_, n_)),
    V_(arma::eye(m_, m_))
  {}

  /** \brief Constructor with given state-space vectors
   *
   * Constructor with given state-space vectors. Initializes all vectors and
   * matrices to have correct dimensions and assigns given vectors. Both
   * predicted an estimated state vectors are assigned with given state vector.
   * All covariance matrices are initialized with ones on diagonal. The rest
   * of data is zero-value initialized.
   *
   * \param u the input vector
   * \param y the output vector
   * \param q the state vector
  */
  ExtendedKalmanFilter(const arma::vec& u, const arma::vec& y, const arma::vec& q) :
    KalmanFilter(u, y, q),
    W_(arma::eye(n_, n_)),
    V_(arma::eye(m_, m_))
  {}


  void setProcessErrorJacobian(const arma::mat W) {
    if (arma::size(W) != arma::size(W_))
      throw std::length_error("Incorrect dimensions");
    else
      W_ = W;
  }

  void setOutputErrorJacobian(const arma::mat V) {
    if (arma::size(V) != arma::size(V_))
      throw std::length_error("Incorrect dimensions");
    else
      V_ = V;
  }

  void setProcessFunction(arma::vec (*process_function)(arma::vec, arma::vec)) {
    processFunction = process_function;
  }

  void setOutputFunction(arma::vec (*output_function)(arma::vec)) {
    outputFunction = output_function;
  }

  virtual void predictState() {
    q_pred_ = processFunction(q_est_, u_);
    P_ = A_ * P_ * trans(A_) + W_ * Q_ * trans(W_);
  }

  virtual void predictState(const arma::vec& u) {
    setInput(u);
    predictState();
  }

  virtual void correctState() {
    S_ = C_ * P_ * trans(C_) + V_ * R_ * trans(V_);
    K_ = P_ * trans(C_) * inv(S_);
    q_est_ = q_pred_ + K_ * (y_ - outputFunction(q_pred_));
    P_ = (I_ - K_ * C_) * P_;
  }

  void correctState(const arma::vec& y) {
    setOutput(y);
    correctState();
  }

protected:
  ExtendedKalmanFilter() {}

  arma::vec (*processFunction)(arma::vec, arma::vec);
  arma::vec (*outputFunction)(arma::vec);

  arma::mat W_;
  arma::mat V_;
};

} // namespace kf
