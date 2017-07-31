# KalmanFilters

A header only c++11 implementation of standard Kalman filter and extended Kalman filter. The code relies heavily on [Armadillo C++](www.arma.sourceforge.net) library for linear algebra operations. Create documentation with `doxygen Doxyfile` and find help in `examples` folder. The examples use [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) to plot the data.

A good introduction to Kalman filter can be found in: www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

## Kalman filter example:

The KF example considers the problem of position estimation for an object moving along 1D line, which control signal is the acceleration (cf. Fig 1). Both the measurement and the control signals are noisy. Additionally, the measurements come in a slower rate than the filter works

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

