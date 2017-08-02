#pragma once

#include <functional>

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
  UnscentedKalmanFilter(uint dim_in, uint dim_out, uint dim_state) :
    ExtendedKalmanFilter(dim_in, dim_out, dim_state)
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
    ExtendedKalmanFilter(A, B, C)
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
    ExtendedKalmanFilter(u, y, q)
  {}


  /** \brief Performs the UKF prediction step
    *
    * TODO
    *
    * \sa processFunction()
   */
  virtual void predictState() {
    q_pred_ = processFunction_(q_est_, u_);
    A_ = processJacobian_(q_est_, u_);
    P_ = A_ * P_ * trans(A_) + Q_;
  }

  /** \brief Performs the UKF prediction step given the input vector
   *
   * TODO
   *
   * \param u is the input vector with dimension l
   * \sa KalmanFilter::setInput(), processFunction_()
  */
  virtual void predictState(const arma::vec& u) {
    setInput(u);
    predictState();
  }

  /** \brief Performs the UKF correction step
   *
   * TODO
   *
   * \sa outputFunction_()
  */
  virtual void correctState() {
    C_ = outputJacobian_(q_est_);
    S_ = C_ * P_ * trans(C_) + R_;
    K_ = P_ * trans(C_) * inv(S_);
    q_est_ = q_pred_ + K_ * (y_ - outputFunction_(q_pred_));
    P_ = (I_ - K_ * C_) * P_;
  }

  /** \brief Performs the UKF correction step given the output vector
   *
   * TODO
   *
   * \sa KalmanFilter::setOutput(), outputFunction_()
  */
  virtual void correctState(const arma::vec& y) {
    setOutput(y);
    correctState();
  }

protected:
  /** \brief This class cannot be instantiated without providing dimensions. */
  UnscentedKalmanFilter() {}

private:
  /** \brief This base class member function is hidden because it serves no use
   * in this context.
  */
  void setProcessJacobian() {}

  /** \brief This base class member function is hidden because it serves no use
   * in this context.
  */
  void setOutputJacobian() {}
};

} // namespace kf
