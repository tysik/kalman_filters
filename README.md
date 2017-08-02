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

Imagine you have an object moving along a 1D line (cf. Fig 1). You have the measurement (output) of its position but it is very noisy and comes with the rate of only 10 Hz. You also know the control signal (input) of this object, which is acceleration. The input (working with the rate of 100 Hz) is also noisy but you would like to take your chance and somehow fuse the measurements with the model.

-----------------------
<p align="center">
  <img src="https://user-images.githubusercontent.com/1482514/28871866-cff75ac4-7786-11e7-8280-237df07c0d60.png" alt="Vehicle with position `x`, velocity `v`, acceleration `a`."/>
  <br/>
  Fig. 1. Vehicle with position `x`, velocity `v`, acceleration `a`.
</p>

-----------------------

Let's start with finding the state space representation of this system. Let the state be composed of position and velocity (`q = [p ; v]`, `n = 2`). The input is acceleration (`u = [a]`, `l = 1`), The output is position measurement (`y = [p]`, `m = 1`). According to state equation `q(k+1) = Aq(k) + Bu(k)` and output equation `y(k) = Cx(k)` state space matrices are:

`A = [1, Tp ; 0, 1]`

`B = [0.5 Tp^2 ; Tp]`

`C = [1, 0]`

Once the KF object is created we have to tune its parameters. Although we could obtain the true measurement covariance (by taking a lot of samples and calculations) we can simply set it to one (`R = 1`) and then tune the process covariance. After all it is the ratio of these two which is the most important. After some testing we found `Q = [0.001, 0 ; 0, 0.001]`, so the prediction is more reliable than update.

During simulation we update the KF at every sample with function `updateState(u, y)` even though the measurement comes only once every ten samples. The reason for that is that this measurement is the latest, most believed measurement we have got so we _freeze_ it. You can see the output of KF estimation on Fig. 2. Estimated position is visually better than pure measurement.

-----------------------
<p align="center">
  <img src="https://user-images.githubusercontent.com/1482514/28871865-cfd7d42e-7786-11e7-9671-b49a67741ccd.png" alt="Exemplary use of KF."/>
  <br/>
  Fig. 2. Exemplary use of KF.
</p>

-----------------------

## Setting up and using the extended Kalman filter

Using EKF is very similar to using KF. You can use `ExtendedKalmanFilter(l, m, n)` constructor just like before to initialize the dimensions or `KalmanFilter(u, y, q_est)` constructor to set the dimensions and initialize the vectors with given values. Constructor `KalmanFilter(A, B, C)` is still valid but is not recommended because matrices `A`, `B` and `C` have different meaning in EKF and will be overwritten with the first update.

Because in EKF the system and output are (usually) nonlinear functions we have to provide such functions, which will recalculate the state and output prediction as well as the state and output Jacobians every time we would like to update the estimate. For that we can use functions `setProcessFunction(pf)`, `setOutputFunction(of)`, `setProcessJacobian(pj)`, and `setOutputJacobian(oj)`, where `pf`, `of`, `pj`, and `oj` are callable objects (like functions, lambdas, member functions, function objects).

Using the EKF is exactly the same as in the case of standard KF. One has to tune matrices `Q` and `R` and call `updateState(u, y)` function.

## Extended Kalman filter example:

Imagine there is a rigid pendulum of lenght `d` with a point-mass `m` attached to its end (cf. Fig. 3). Furthermore, the bearing of the pendulum is not perfect because it creates viscous friction of coefficient `b`. For some reason, somebody attached this pendulum to a frame moving in a vertical direction. We can apply acceleration `a` to this frame in order to control it. We would like to know the exact angular position of the pendulum, but we only have the measurements of `x` and `z` coordinates of the pendulum end-point, described w.r.t. moving frame. The rate of the measurements is 20 Hz and it is noisy. The input is also noisy but we would like to use it for filtration.

-----------------------
<p align="center">
  <img src="https://user-images.githubusercontent.com/1482514/28871962-3c552dcc-7787-11e7-9cb5-31b6c049324d.png" alt="Pendulum with point-mass `m`, length `d`, friction coef. `b`, attached to a frame moving with acceleration `a`."/>
  <br/>
  Fig. 3. Pendulum with point-mass `m`, length `d`, friction coef. `b`, attached to a frame moving with acceleration `a`.
</p>

-----------------------

First let's start with state-space representation. Let the state be composed of angular position and velocity (`q = [phi ; omega]`, `n = 2`). The input is acceleration of the moving frame (`u = [a]`, `l = 1`). The output is the measurement of end-point coordinates (`y = [x ; z]`, `m = 2`). Now let's define the discrete process and output functions:

`q(k+1) = f(q(k), u(k)) = [phi(k) + omega(k) Tp ; omega(k) + ( m d (g + a(k)) sin(phi(k)) - b omega(k) ) Tp / (m d^2) ]`

`y(k) = g(q(k)) = [d sin(phi(k)) ; d cos(phi(k))]`

To get the linear part for proper covariance propagation, we need to find the Jacobians of these function w.r.t. state space:

`df / dq = [1, Tp ; m d (g + a(k)) cos(phi(k)) Tp / (m d^2), 1 - b Tp / (m d^2) ]`

`dg / dq = [d cos(phi(k)), 0 ; -d sin(phi(k)), 0 ]`

Once the system equations are set, it is time to tune the parameters. As in the KF example, we set measurement covariance to one (`R = [1, 0 ; 0, 1]`). After some testing we found again `Q = [0.001, 0 ; 0, 0.001]`. The results can be seen on Fig. 4.

-----------------------
<p align="center">
  <img src="https://user-images.githubusercontent.com/1482514/28871961-3c53bf14-7787-11e7-832b-8425597d2691.png" alt="Exemplary use of EKF."/>
  <br/>
  Fig. 4. Exemplary use of EKF.
</p>

-----------------------

