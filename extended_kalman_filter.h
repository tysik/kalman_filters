#pragma once

#include <functional>

#include "kalman_filter.h"

namespace kf {

/** \class ExtendedKalmanFilter extended_kalman_filter.h
 * \brief Class containing functionality of extended Kalman filter
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
class ExtendedKalmanFilter : public KalmanFilter
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
  ExtendedKalmanFilter(size_t dim_in, size_t dim_out, size_t dim_state) :
    KalmanFilter(dim_in, dim_out, dim_state)
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
  ExtendedKalmanFilter(const arma::mat& A, const arma::mat& B, const arma::mat& C) :
    KalmanFilter(A, B, C)
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
  ExtendedKalmanFilter(const arma::vec& u, const arma::vec& y, const arma::vec& q) :
    KalmanFilter(u, y, q)
  {}


  /** \brief Process function setter
   *
   * \param processFunction is a callable object (e.g. function or lambda)
   *                        representing the process function
   * \sa processFunction_
  */
  void setProcessFunction(std::function<arma::vec(const arma::vec& q,
                                                  const arma::vec& u)> processFunction) {
    processFunction_ = processFunction;
  }

  /** \brief Output function setter
   *
   * \param outputFunction is a callable object (e.g. function or lambda)
   *                       representing the output function
   * \sa outputFunction_
  */
  void setOutputFunction(std::function<arma::vec(const arma::vec& q)> outputFunction) {
    outputFunction_ = outputFunction;
  }

  /** \brief Process Jacobian setter
   *
   * \param processJacobian is a callable object (e.g. function or lambda)
   *                        representing the process Jacobian function
   * \sa processJacobian_
  */
  void setProcessJacobian(std::function<arma::mat(const arma::vec& q,
                                                  const arma::vec& u)> processJacobian) {
    processJacobian_ = processJacobian;
  }

  /** \brief Output Jacobian setter
   *
   * \param outputJacobian is a callable object (e.g. function or lambda)
   *                        representing the output Jacobian function
   * \sa outputJacobian_
  */
  void setOutputJacobian(std::function<arma::mat(const arma::vec& q)> outputJacobian) {
    outputJacobian_ = outputJacobian;
  }


  /** \brief Performs the EKF prediction step
    *
    * Calculates the state evolution for the current time step based on previous
    * state estimate and input. Updates the process Jacobian and estimate error
    * covariance matrix.
    *
    * \sa processFunction_, processJacobian_
   */
  virtual void predictState() {
    q_pred_ = processFunction_(q_est_, u_);
    A_ = processJacobian_(q_est_, u_);
    P_ = A_ * P_ * trans(A_) + Q_;
  }


  /** \brief Performs the EKF correction step
   *
   * Calculates the new Kalman gain, corrects the state prediction to obtain new
   * state estimate, and updates estimate error covariance as well as innovation
   * covariance.
   *
   * \sa outputFunction_, outputJacobian_
  */
  virtual void correctState() {
    C_ = outputJacobian_(q_est_);
    S_ = C_ * P_ * trans(C_) + R_;
    K_ = P_ * trans(C_) * inv(S_);
    q_est_ = q_pred_ + K_ * (y_ - outputFunction_(q_pred_));
    P_ = (I_ - K_ * C_) * P_;
  }


protected:

  /** \brief This class cannot be instantiated without providing dimensions. */
  ExtendedKalmanFilter() {}

  /** \brief Process function
   *
   * Wrapper of a function which returns arma::vec of dimension n, representing
   * the new state vector, and which takes two arguments of type arma::vec, with
   * dimensions n and l, respectively. The arguments of the function are
   * the state vector and input vector, respectively.
   *
   * The process function f() describes the evolution of the discrete system
   * based on previous state and input vectors.
   *
   * \param q is the state vector from the previous time step
   * \param u is the input vector from the previous time step
   *
   * \returns the state vector for the current time step
   *
   * \sa setProcessFunction()
  */
  std::function<arma::vec(const arma::vec& q, const arma::vec& u)> processFunction_;

  /** \brief Output function
   *
   * Wrapper of a function which returns arma::vec of dimension m, representing
   * the output vector, and taking single argument of type arma::vec, with
   * dimension n, representing the state vector.
   *
   * The output function describes the transformation of state into output
   * dimension.
   *
   * \param q is the state vector for the current time step
   *
   * \returns the output vector for the current time step
   *
   * \sa setOutputFunction()
  */
  std::function<arma::vec(const arma::vec& q)> outputFunction_;

  /** \brief Process Jacobian
   *
   * Wrapper of a function which returns arma::mat of dimensions n x n,
   * representing the linear approximation of the process function f(). The
   * arguments of the function are the state and input vectors from the previous
   * time step.
   *
   * The Jacobian A of process function f() is responsible for proper
   * propagation of process covariance.
   *
   * \param q is the state vector from the previous time step
   * \param u is the input vector from the previous time step
   *
   * \returns the Jacobian A of process function f()
   *
   * \sa processFunction_, setProcessFunction()
  */
  std::function<arma::mat(const arma::vec& q, const arma::vec& u)> processJacobian_;

  /** \brief Output Jacobian
   *
   * Wrapper of a function which returns arma::mat of dimensions m x n,
   * representing the linear approximation of the output function h(). The
   * argument of the function is the state vector for the current time step.
   *
   * The Jacobian C of output function h() is responsible for proper propagation
   * of measurement covariance.
   *
   * \param q is the state vector for the current time step
   *
   * \returns the Jacobian C of output function h()
   *
   * \sa outputFunction_, setOutputFunction()
  */
  std::function<arma::mat(const arma::vec& q)> outputJacobian_;
};

} // namespace kf
