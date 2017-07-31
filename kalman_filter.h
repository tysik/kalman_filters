#pragma once

#include <armadillo>
#include <stdexcept>

namespace kf {

/** \class KalmanFilter kalman_filter.h
 * \brief Class containing functionality of Kalman filter.
 *
 * Class containing functionality of Kalman filter. Both prediction and
 * correction steps can be invoked separately.
*/
class KalmanFilter
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
  KalmanFilter(uint dim_in, uint dim_out, uint dim_state) :
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
   * Constructor with given state-space matrices. Initializes all vectors and
   * matrices to have correct dimensions and assigns given state-space matrices.
   * All covariance matrices are initialized with ones on diagonal. The rest
   * of data is zero-value initialized.
   *
   * \param A the state matrix
   * \param B the input matrix
   * \param C the output matrix
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
  KalmanFilter(const arma::vec& u, const arma::vec& y, const arma::mat& q) :
    KalmanFilter(u.n_elem, y.n_elem, q.n_elem)
  {
    u_ = u;
    y_ = y;
    q_pred_ = q;
    q_est_ = q;
  }


  void setStateMatrix(const arma::mat& A) {
    if (arma::size(A) != arma::size(A_))
      throw std::length_error("Incorrect dimensions");
    else
      A_ = A;
  }

  void setInputMatrix(const arma::mat& B) {
    if (arma::size(B) != arma::size(B_))
      throw std::length_error("Incorrect dimensions");
    else
      B_ = B;
  }

  void setOutputMatrix(const arma::mat& C) {
    if (arma::size(C) != arma::size(C_))
      throw std::length_error("Incorrect dimensions");
    else
      C_ = C;
  }

  void setEstimateCovariance(const arma::mat& P) {
    if (arma::size(P) != arma::size(P_))
      throw std::length_error("Incorrect dimensions");
    else
      P_ = P;
  }

  void setProcessCovariance(const arma::mat& Q) {
    if (arma::size(Q) != arma::size(Q_))
      throw std::length_error("Incorrect dimensions");
    else
      Q_ = Q;
  }

  void setOutputCovariance(const arma::mat& R) {
    if (arma::size(R) != arma::size(R_))
      throw std::length_error("Incorrect dimensions");
    else
      R_ = R;
  }


  void setInput(const arma::vec& u) {
    if (arma::size(u) != arma::size(u_))
      throw std::length_error("Incorrect dimensions");
    else
      u_ = u;
  }

  void setOutput(const arma::vec& y) {
    if (arma::size(y) != arma::size(y_))
      throw std::length_error("Incorrect dimensions");
    else
      y_ = y;
  }

  void setPrediction(const arma::vec& q_pred) {
    if (arma::size(q_pred) != arma::size(q_pred_))
      throw std::length_error("Incorrect dimensions");
    else
      q_pred_ = q_pred;
  }

  void setEstimate(const arma::vec& q_est) {
    if (arma::size(q_est) != arma::size(q_est_))
      throw std::length_error("Incorrect dimensions");
    else
      q_est_ = q_est;
  }


  const arma::mat& getEstimateCovariance() const { return P_; }

  const arma::mat& getInnovationCovariance() const { return S_; }

  const arma::mat& getKalmanGain() const { return K_; }


  const arma::vec& getInput() const { return u_; }

  const arma::vec& getOutput() const { return y_; }

  const arma::vec& getPrediction() const { return q_pred_; }

  const arma::vec& getEstimate() const { return q_est_; }


  /** \brief Performs the KF prediction step.
   *
   * Performs the KF prediction step: calulates the state predicted from the
   * model and updates the estimate error covariance matrix.
  */
  virtual void predictState() {
    q_pred_ = A_ * q_est_ + B_ * u_;
    P_ = A_ * P_ * trans(A_) + Q_;
  }

  /** \brief Performs the KF prediction step given input u.
   *
   * Performs the KF prediction step given input u: calulates the state
   * predicted from the model and updates the estimate error covariance matrix.
   *
   * \param u the input vector
   * \sa setInput(), predictState()
  */
  virtual void predictState(const arma::vec& u) {
    setInput(u);
    predictState();
  }

  /** \brief Performs the KF correction step.
   *
   * Performs the KF correction step: calulates the new Kalman gain, corrects
   * the state prediction to obtain new state estimate, and updates estimate
   * error covariance as well as innovation covariance.
  */
  virtual void correctState() {
    S_ = C_ * P_ * trans(C_) + R_;
    K_ = P_ * trans(C_) * inv(S_);
    q_est_ = q_pred_ + K_ * (y_ - C_ * q_pred_);
    P_ = (I_ - K_ * C_) * P_;
  }

  /** \brief Performs the KF correction step given output y.
   *
   * Performs the KF correction step given output y: calulates the new Kalman
   * gain, corrects the state prediction to obtain new state estimate, and
   * updates estimate error covariance as well as innovation covariance.
   *
   * \param y the output (measurement) vector
   * \sa setOutput(), correctState()
  */
  void correctState(const arma::vec& y) {
    setOutput(y);
    correctState();
  }

  /** \brief Performs the KF update.
   *
   * Performs the KF update, i.e. both prediction and correction steps.
   *
   * * \sa predictState(), correctState()
  */
  void updateState() {
    predictState();
    correctState();
  }

  /** \brief Performs the KF update given input u and output y.
   *
   * Performs the KF update given input u and output y, i.e. sets the new values
   * and executes both prediction and correction steps.
   *
   * \param u the input vector
   * \param y the output (measurement) vector
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
  uint l_;            /**< Dimension of input vector (number of control signals).
                           Can be zero for autonomous systems. */
  uint m_;            /**< Dimension of output vector (number of measured values) */
  uint n_;            /**< Dimension of state vector */

  // System matrices:
  arma::mat A_;       /**< State matrix with dimensions n x n */
  arma::mat B_;       /**< Input matrix with dimensions n x l */
  arma::mat C_;       /**< Output matrix with dimensions m x n */

  // Kalman gain matrix:
  arma::mat K_;       /**< Kalman gain; matrix with dimensions n x m */

  // Identity matrix
  arma::mat I_;       /**< Identity matrix with dimensions n x n */

  // Covariance matrices:
  arma::mat P_;       /**< Estimate covariance matrix with dimensions n x n */
  arma::mat Q_;       /**< Process covariance matrix with dimensions n x n */
  arma::mat R_;       /**< Measurement covariance matrix with dimensions m x m */
  arma::mat S_;       /**< Innovation covariance matrix with dimensions m x m */

  // Signals:
  arma::vec u_;       /**< Input vector with dimension l */
  arma::vec y_;       /**< Output vector with dimension m */
  arma::vec q_pred_;  /**< Predicted state vector with dimension n */
  arma::vec q_est_;   /**< Estimated state vector with dimension n */
};

} // namespace kf
