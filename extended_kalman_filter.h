#pragma once

#include "kalman_filter.h"

namespace kf {

/** \class ExtendedKalmanFilter extended_kalman_filter.h
 * \brief Class containing functionality of extended Kalman filter.
 *
 * Throughout the description "n" denotes dimension of state vector, "m" denotes
 * dimension of output vector (measured values), and "l" denotes dimension of
 * input vector (control signals). Both prediction and correction steps can be
 * invoked separately.
*/
class ExtendedKalmanFilter : public KalmanFilter
{
public:

  /** \brief Constructor with given dimensions.
   *
   * Initializes all vectors and matrices to have correct dimensions. State
   * matrix as well as all covariance matrices are initialized with ones on
   * diagonal. The rest of data is zero-value initialized.
   *
   * \param dim_in is the dimension of input vector (number of control signals).
   * \param dim_out is the dimension of output vector (number of measurements).
   * \param dim_state is the dimension of state vector.
  */
  ExtendedKalmanFilter(uint dim_in, uint dim_out, uint dim_state) :
    KalmanFilter(dim_in, dim_out, dim_state),
    W_(arma::eye(n_, n_)),
    V_(arma::eye(m_, m_))
  {}

  /** \brief Constructor with given state-space matrices
   *
   * Initializes all vectors and matrices to have correct dimensions and assigns
   * given state-space matrices. All covariance matrices are initialized with
   * ones on diagonal. The rest of data is zero-value initialized.
   *
   * \param A is the state matrix.
   * \param B is the input matrix.
   * \param C is the output matrix.
  */
  ExtendedKalmanFilter(const arma::mat& A, const arma::mat& B, const arma::mat& C) :
    KalmanFilter(A, B, C),
    W_(arma::eye(n_, n_)),
    V_(arma::eye(m_, m_))
  {}

  /** \brief Constructor with given state-space vectors
   *
   * Initializes all vectors and matrices to have correct dimensions and assigns
   * given vectors. Both predicted an estimated state vectors are assigned with
   * given state vector. All covariance matrices are initialized with ones on
   * diagonal. The rest of data is zero-value initialized.
   *
   * \param u is the input vector.
   * \param y is the output vector.
   * \param q is the state vector.
  */
  ExtendedKalmanFilter(const arma::vec& u, const arma::vec& y, const arma::vec& q) :
    KalmanFilter(u, y, q),
    W_(arma::eye(n_, n_)),
    V_(arma::eye(m_, m_))
  {}


  /** \brief Process error Jacobian setter
   *
   * \param W is the process error Jacobian with dimension n x n.
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions.
  */
  void setProcessErrorJacobian(const arma::mat W) {
    if (arma::size(W) != arma::size(W_))
      throw std::length_error("Incorrect dimensions");
    else
      W_ = W;
  }

  /** \brief Output error Jacobian setter
   *
   * \param V is the output error Jacobian with dimension m x m.
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions.
  */
  void setOutputErrorJacobian(const arma::mat V) {
    if (arma::size(V) != arma::size(V_))
      throw std::length_error("Incorrect dimensions");
    else
      V_ = V;
  }


  /** \brief Process function setter
   *
   * \param processFunction is the pointer to the process function.
   *
   * \sa processFunction()
  */
  void setProcessFunction(arma::vec (*process_function)(arma::vec, arma::vec)) {
    processFunction = process_function;
  }

  /** \brief Output function setter
   *
   * \param outputFunction is the pointer to the output function.
   *
   * \sa outputFunction()
  */
  void setOutputFunction(arma::vec (*output_function)(arma::vec)) {
    outputFunction = output_function;
  }


  /** \brief Performs the EKF prediction step.
   *
   * Calulates the state evolution for the current time step based on previous
   * state estimate and input. Updates the estimate error covariance matrix
   * based on given process Jacobian and process error Jacobian.
   *
   * \sa processFunction()
  */
  virtual void predictState() {
    q_pred_ = processFunction(q_est_, u_);
    P_ = A_ * P_ * trans(A_) + W_ * Q_ * trans(W_);
  }

  /** \brief Performs the EKF prediction step given input vector.
   *
   * Sets provided input vector.
   * Calulates the state evolution for the current time step based on previous
   * state estimate and input. Updates the estimate error covariance matrix
   * based on given process Jacobian and process error Jacobian.
   *
   * \param u is the input vector with dimension l.
   *
   * \sa KalmanFilter::setInput(), processFunction()
  */
  virtual void predictState(const arma::vec& u) {
    setInput(u);
    predictState();
  }

  /** \brief Performs the EKF correction step.
   *
   * Calulates the state evolution for the current time step based on previous
   * state estimate and input. Updates the estimate error covariance matrix
   * based on given process Jacobian and process error Jacobian.
   *
   * \sa processFunction()
  */
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
  /** \brief This class cannot be instantiated without providing dimensions. */
  ExtendedKalmanFilter() {}

  /** \brief Process function
   *
   * Pointer to a function which returns arma::vec of dimension n, representing
   * the new state vector, and which takes two arguments of type arma::vec, with
   * dimensions n and l, respectively. The arguments of the function are
   * previous state vector and input vector.
   *
   * The process function describes the evolution of the discrete system based
   * on previous state and input vectors.
   *
   * \param 1 is the state vector from the previous time step.
   * \param 2 is the input vector from the previous time step.
   *
   * \returns the state vector for the current time step.
   *
   * \sa setProcessFunction()
  */
  arma::vec (*processFunction)(arma::vec, arma::vec);

  /** \brief Output function
   *
   * Pointer to a function which returns arma::vec of dimension m, representing
   * the output vector, and taking single argument of type arma::vec, with
   * dimension n, representing the state vector.
   *
   * The output function describes the transformation of state into output
   * dimension.
   *
   * \param 1 is the state vector for the current time step.
   *
   * \returns the output vector for the current time step.
   *
   * \sa setOutputFunction()
  */
  arma::vec (*outputFunction)(arma::vec);

  arma::mat W_;     /**< Process error Jacobian with dimensions n x n */
  arma::mat V_;     /**< Output error Jacobian with dimensions m x m */
};

} // namespace kf
