# KalmanFilters

A header only c++11 implementation of standard Kalman filter and extended Kalman filter. The code relies heavily on [Armadillo C++](www.arma.sourceforge.net) library for linear algebra operations. Create documentation with `doxygen Doxyfile` and find help in `examples` folder. The examples use [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) to plot the data.

A good introduction to Kalman filter can be found in: www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

## Setting up and using the Kalman filter

The first step to prepare the KF object is to recognize the size of state (`n` = how many variables we want to estimate), input (`l` = how many control signals does the system use), and output (`m` = how many measured values do we acquire). At this point we can already initialize the KF object with constructor `KalmanFilter(l, m, n)`. The constructor will prepare all the matrices with given dimensions. 

Next step is to write down the discrete state, input and output matrices for the system (i.e. A, B, and C). We can set them with functions `setStateMatrix(A)`, `setInputMatrix(B)`, `setOutputMatrix(C)`. Other possibility is to directly use constructor `KalmanFilter(A, B, C)` which will induce the dimensions from the matrices. 

We might also want to set some initial values for input, output, and predicted/estimated state vectors. For that we will use functions `setInput(u)`, `setOutput(y)`, `setPrediction(q_pred)`, `setEstimate(q_est)` (where `u`, `y`, `q_pred`, `q_est` are the input, output, predicted state and estimated state vectors). We can also use constructor `KalmanFilter(u, y, q_est)` which will deduce the dimensions from the vectors. This constructor will assign value of `q_est` to the variable `q_pred`.

Once the system matrices and vectors are set it is time to tune the KF. The tunable parameters of KF are the process and output covariance matrices: `Q` and `R`. Once their values are obtained, we can set them with functions `setProcessCovariance(Q)` and `setOutputCovariance(R)`.

Now it's time for action. If the measurements come with the same rate as the system loop it is recommended to use function `updateState(u, y)` with most recent input signal `u` and output signal `y`. If the rates are different we can separately use functions `predictState()` (`predictState(u)`) and `correctState()` (`correctState(y)`) which will perform the KF steps with previous or with given input/output.

To get the latest available state estimate use `getEstimate()` function.


## Kalman filter example:

Provided KF example considers the problem of position estimation for an object moving along 1D line, which control signal is the acceleration (cf. Fig 1). Both the measurement and the control signals are noisy. Additionally, the measurements come in a slower rate than the filter works.

-----------------------
<p align="center">
  <img src="https://user-images.githubusercontent.com/1482514/28799173-c7996088-7647-11e7-910c-6f1006ca3659.png" alt="Linear system."/>
  <br/>
  Fig. 1. Linear system.
</p>

-----------------------


-----------------------
<p align="center">
  <img src="https://user-images.githubusercontent.com/1482514/28791866-0c439e0c-762e-11e7-8ee6-cac6ed5bf844.png" alt="Exemplary use of KF."/>
  <br/>
  Fig. 2. Exemplary use of KF.
</p>

-----------------------

## Setting up and using the Kalman filter

Using EKF is very similar to using KF. You can use `ExtendedKalmanFilter(l, m, n)` constructor just like before to initialize the dimensions or `KalmanFilter(u, y, q_est)` constructor to set the dimensions and initialize the vectors with given values. Construct `KalmanFilter(A, B, C)` is still valid but is not recommended because matrices `A`, `B` and `C` have different meaning in EKF and will be overwritten with the first update.

Because in EKF the system and output are (usually) nonlinear functions we have to provide such functions, which will recalculate the state and output prediction as well as the state and output Jacobians every time we would like to update the estimate. For that we can use functions `setProcessFunction(pf)`, `setOutputFunction(of)`, `setProcessJacobian(pj)`, and `setOutputJacobian(oj)`, where `pf`, `of`, `pj`, and `oj` are callable objects (like functions, lambdas, member functions, function objects).

Using the EKF is exactly the same as in the case of standard KF. One has to prepare the tunned `Q` and `R` matrices and call `updateState(u, y)` function.

## Extended Kalman filter example:

The EKF example considers the problem of angle estimation for a pendulum attached to a platform moving in a vertical direction (cf. Fig. 3). The measurements are (x,y) coordinates of the pendulum end point (taken in the moving frame). The control signal of the system is acceleration of the platform.

-----------------------
<p align="center">
  <img src="https://user-images.githubusercontent.com/1482514/28799174-c79f8e5e-7647-11e7-97af-f6754a174e13.png" alt="Nonlinear system."/>
  <br/>
  Fig. 3. Nonlinear system.
</p>

-----------------------


-----------------------
<p align="center">
  <img src="https://user-images.githubusercontent.com/1482514/28791872-0f9a9858-762e-11e7-984e-bc7f57e2fa4e.png" alt="Exemplary use of EKF."/>
  <br/>
  Fig. 4. Exemplary use of EKF.
</p>

-----------------------

