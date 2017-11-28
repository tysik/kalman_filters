#pragma once

#include <armadillo>
#include <stdexcept>

/*! \mainpage Kalman Filters
 *
 * kf::KalmanFilter
 *
 * kf::ExtendedKalmanFilter
 *
 * kf::UnscentedKalmanFilter
 */

namespace kf {

/** \class KalmanFilter kalman_filter.h
 * \brief Class containing functionality of the Kalman filter
 *
 * The process is described with equation q(k) = A * q(k-1) + B * u(k-1)
 * The output is described with equation y(k) = C * q(k)
 *
 * Throughout the description "n" denotes dimension of state vector q, "m"
 * denotes dimension of output vector y (measured values), and "l" denotes
 * dimension of input vector u (control signals). Both prediction and correction
 * steps can be invoked separately.
 *
 * The class exploits Armadillo library for matrix operations and can throw any
 * of its exceptions (cf. www.arma.sourceforge.net).
*/
class KalmanFilter
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
  KalmanFilter(size_t dim_in, size_t dim_out, size_t dim_state) :
    l_(dim_in),
    m_(dim_out),
    n_(dim_state),

    A_(arma::eye(n_, n_)),
    B_(arma::zeros(n_, l_)),
    C_(arma::zeros(m_, n_)),

    K_(arma::zeros(n_, m_)),
    I_(arma::eye(n_, n_)),

    P_(arma::eye(n_, n_)),
    Q_(arma::eye(n_, n_)),
    R_(arma::eye(m_, m_)),
    S_(arma::eye(m_, m_)),

    u_(arma::zeros(l_)),
    y_(arma::zeros(m_)),
    q_pred_(arma::zeros(n_)),
    q_est_(arma::zeros(n_))
  { }

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
  KalmanFilter(const arma::mat& A, const arma::mat& B, const arma::mat& C) :
    KalmanFilter(B.n_cols, C.n_rows, A.n_rows)
  {
    A_ = A;
    B_ = B;
    C_ = C;
  }

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
  KalmanFilter(const arma::vec& u, const arma::vec& y, const arma::mat& q) :
    KalmanFilter(u.n_elem, y.n_elem, q.n_elem)
  {
    u_ = u;
    y_ = y;
    q_pred_ = q;
    q_est_ = q;
  }


  /** \brief State matrix setter
   *
   * \param A is the state matrix with dimensions n x n
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions
  */
  void setStateMatrix(const arma::mat& A) {
    if (arma::size(A) != arma::size(A_))
      throw std::length_error("Incorrect dimensions of state matrix");
    else
      A_ = A;
  }

  /** \brief Input matrix setter
   *
   * \param B is the input matrix with dimensions n x l
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions
  */
  void setInputMatrix(const arma::mat& B) {
    if (arma::size(B) != arma::size(B_))
      throw std::length_error("Incorrect dimensions of input matrix");
    else
      B_ = B;
  }

  /** \brief Output matrix setter
   *
   * \param C is the output matrix with dimensions m x n
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions
  */
  void setOutputMatrix(const arma::mat& C) {
    if (arma::size(C) != arma::size(C_))
      throw std::length_error("Incorrect dimensions of output matrix");
    else
      C_ = C;
  }

  /** \brief Estimate covariance matrix setter
   *
   * \param P is the estimate covariance matrix with dimensions n x n
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions
  */
  void setEstimateCovariance(const arma::mat& P) {
    if (arma::size(P) != arma::size(P_))
      throw std::length_error("Incorrect dimensions of estimate cov. matrix");
    else
      P_ = P;
  }

  /** \brief Process covariance matrix setter
   *
   * \param Q is the process covariance matrix with dimensions n x n
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions
  */
  void setProcessCovariance(const arma::mat& Q) {
    if (arma::size(Q) != arma::size(Q_))
      throw std::length_error("Incorrect dimensions of process cov. matrix");
    else
      Q_ = Q;
  }

  /** \brief Output covariance matrix setter
   *
   * \param R is the output covariance matrix with dimensions m x m
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions
  */
  void setOutputCovariance(const arma::mat& R) {
    if (arma::size(R) != arma::size(R_))
      throw std::length_error("Incorrect dimensions of output cov. matrix");
    else
      R_ = R;
  }


  /** \brief Input vector setter
   *
   * \param u is the input vector with dimension l
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions
  */
  void setInput(const arma::vec& u) {
    if (arma::size(u) != arma::size(u_))
      throw std::length_error("Incorrect dimensions of input vector");
    else
      u_ = u;
  }

  /** \brief Output vector setter
   *
   * \param y is the output vector with dimension m
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions
  */
  void setOutput(const arma::vec& y) {
    if (arma::size(y) != arma::size(y_))
      throw std::length_error("Incorrect dimensions of output vector");
    else
      y_ = y;
  }

  /** \brief Predicted state vector setter
   *
   * \param q_pred is the predicted state vector with dimension n
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions
  */
  void setPrediction(const arma::vec& q_pred) {
    if (arma::size(q_pred) != arma::size(q_pred_))
      throw std::length_error("Incorrect dimensions of predicted state vector");
    else
      q_pred_ = q_pred;
  }

  /** \brief Estimated state vector setter
   *
   * \param q_est is the estimated state vector with dimension n
   * \throws std::length_error if the input matrix dimensions are different than
   *                           initially provided dimensions
  */
  void setEstimate(const arma::vec& q_est) {
    if (arma::size(q_est) != arma::size(q_est_))
      throw std::length_error("Incorrect dimensions of estimated state vector");
    else
      q_est_ = q_est;
  }


  /** \brief Estimate covariance matrix getter
   *
   * \returns the current estimate covariance matrix P
  */
  const arma::mat& getEstimateCovariance() const { return P_; }

  /** \brief Innovation covariance matrix getter
   *
   * Innovation covariance matrix is calculated during correction step
   * S = C * P * trans(C) + R.
   *
   * \returns the current innovation covariance matrix S
  */
  const arma::mat& getInnovationCovariance() const { return S_; }

  /** \brief Kalman gain getter
   *
   * Kalman gain is calculated during correction step K = P * trans(C) * inv(S).
   *
   * \returns the current Kalman gain K
  */
  const arma::mat& getKalmanGain() const { return K_; }


  /** \brief Input vector getter
   *
   * \returns the current input vector u
  */
  const arma::vec& getInput() const { return u_; }

  /** \brief Output vector getter
   *
   * \returns the current output vector y
  */
  const arma::vec& getOutput() const { return y_; }

  /** \brief Predicted state vector getter
   *
   * \returns the current predicted state vector q_pred
  */
  const arma::vec& getPrediction() const { return q_pred_; }

  /** \brief Estimated state vector getter
   *
   * \returns the current estimated state vector q_est
  */
  const arma::vec& getEstimate() const { return q_est_; }


  /** \brief Performs the KF prediction step
   *
   * Calculates the state predicted from the model and updates the estimate
   * error covariance matrix.
  */
  virtual void predictState() {
    q_pred_ = A_ * q_est_ + B_ * u_;
    P_ = A_ * P_ * trans(A_) + Q_;
  }

  /** \brief Performs the KF prediction step given input u
   *
   * Sets provided input vector and calculates the state predicted from the
   * model and updates the estimate error covariance matrix.
   *
   * \param u is the input vector
   * \sa setInput(), predictState()
  */
  void predictState(const arma::vec& u) {
    setInput(u);
    predictState();
  }

  /** \brief Performs the KF correction step
   *
   * Calculates the new Kalman gain, corrects the state prediction to obtain new
   * state estimate, and updates estimate error covariance as well as innovation
   * covariance.
  */
  virtual void correctState() {
    S_ = C_ * P_ * trans(C_) + R_;
    K_ = P_ * trans(C_) * inv(S_);
    q_est_ = q_pred_ + K_ * (y_ - C_ * q_pred_);
    P_ = (I_ - K_ * C_) * P_;
  }

  /** \brief Performs the KF correction step given output y
   *
   * Sets provided output vector and calculates the new Kalman gain. Corrects
   * the state prediction to obtain new state estimate, and updates estimate
   * error covariance as well as innovation covariance.
   *
   * \param y is the output vector
   * \sa setOutput(), correctState()
  */
  void correctState(const arma::vec& y) {
    setOutput(y);
    correctState();
  }

  /** \brief Performs the KF update
   *
   * Executes both prediction and correction steps.
   *
   * \sa predictState(), correctState()
  */
  void updateState() {
    predictState();
    correctState();
  }

  /** \brief Performs the KF update given input u and output y
   *
   * Sets the new values of input and output vectors and executes both
   * prediction and correction steps.
   *
   * \param u is the input (controls) vector
   * \param y is the output (measurement) vector
   * \sa setInput(), setOutput(), predictState(), correctState()
  */
  void updateState(const arma::vec& u, const arma::vec& y) {
    setInput(u);
    setOutput(y);
    updateState();
  }


protected:

  /** \brief This class cannot be instantiated without providing dimensions. */
  KalmanFilter() {}

  // Dimensions:
  size_t l_;            /**< \brief Dimension of input vector (number of control
                           signals). Can be zero for autonomous systems. */
  size_t m_;            /**< \brief Dimension of output vector (number of
                           measured values). Can be zero for prediction only. */
  size_t n_;            /**< \brief Dimension of state vector */

  // System matrices:
  arma::mat A_;       /**< \brief State matrix with dimensions n x n */
  arma::mat B_;       /**< \brief Input matrix with dimensions n x l */
  arma::mat C_;       /**< \brief Output matrix with dimensions m x n */

  // Kalman gain matrix:
  arma::mat K_;       /**< \brief Kalman gain; matrix with dimensions n x m */

  // Identity matrix
  arma::mat I_;       /**< \brief Identity matrix with dimensions n x n */

  // Covariance matrices:
  arma::mat P_;       /**< \brief Estimate covariance matrix with dimensions
                           n x n */
  arma::mat Q_;       /**< \brief Process covariance matrix with dimensions
                           n x n */
  arma::mat R_;       /**< \brief Measurement covariance matrix with dimensions
                           m x m */
  arma::mat S_;       /**< \brief Innovation covariance matrix with dimensions
                           m x m */

  // Signals:
  arma::vec u_;       /**< \brief Input vector with dimension l */
  arma::vec y_;       /**< \brief Output vector with dimension m */
  arma::vec q_pred_;  /**< \brief Predicted state vector with dimension n */
  arma::vec q_est_;   /**< \brief Estimated state vector with dimension n */
};

} // namespace kf
