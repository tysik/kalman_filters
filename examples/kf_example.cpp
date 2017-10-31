/*
 * In this example we show how to estimate the position of a 1D moving object
 * given measurements of its position and control signals in a form of object
 * acceleration. Both measurement and control signals are noisy. The
 * measurements are obtained with ten times smaller rate than the system.
 *
 * The state of the system is q = (p, v) (position and velocity): n = 2
 * The input of the system is u = a (acceleration): l = 1
 * The output of the system is p (position): m = 1
 *
 * The system is described with equation q(k) = A * q(k-1) + B * u(k-1)
 * The output is described with equation y(k) = C * q(k)
 */

#include <iostream>
#include <vector>
#include <random>

#include "../kalman_filter.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using namespace arma;
using namespace std;
using namespace kf;

// Time and samples
const double system_dt = 0.01;        // The system works with rate of 100 Hz
const double measurement_dt = 0.1;    // The measurements come with rate of 10 Hz
const double simulation_time = 10.0;  // The time of simulation is 10 s

const int N = static_cast<int>(simulation_time / system_dt);
const int M = static_cast<int>(measurement_dt / system_dt);

int main() {
  // Buffers for plots
  vector<double> time(N);

  vector<double> true_pos(N);
  vector<double> true_vel(N);
  vector<double> true_acc(N);

  vector<double> measured_pos(N);
  vector<double> estimated_pos(N);
  vector<double> believed_acc(N);

  // Pseudo random numbers generator
  double measurement_mu = 0.0;      // Mean
  double measurement_sigma = 0.5;   // Standard deviation

  double process_mu = 0.0;
  double process_sigma = 0.05;

  default_random_engine generator;
  normal_distribution<double> measurement_noise(measurement_mu, measurement_sigma);
  normal_distribution<double> process_noise(process_mu, process_sigma);

  // Preparing KF
  mat A = { {1.0, system_dt},
            {0.0, 1.0      } };

  mat B = vec({system_dt * system_dt / 2.0, system_dt});

  mat C = {1.0, 0.0};

  KalmanFilter kf(A, B, C);

  // The process and measurement covariances are sort of tunning parameters
  mat Q = {{0.001, 0.0}, {0.0, 0.001}};
  mat R = {1.0};

  kf.setProcessCovariance(Q);
  kf.setOutputCovariance(R);

  // Initial values (unknown by KF)
  time[0] = 0.0;
  true_pos[0] = 1.0;
  true_vel[0] = 0.1;
  true_acc[0] = 0.0;

  measured_pos[0] = 0.0;
  estimated_pos[0] = 0.0;
  believed_acc[0] = 0.0;

  // Simulation
  for (size_t i = 1; i < N; ++i) {
    time[i] = i * system_dt;

    // We belive that acceleration was this
    believed_acc[i] = sin(time[i] * 2.0 * M_PI / simulation_time);

    // In fact there was some noise on input
    true_acc[i] = believed_acc[i] + process_noise(generator);
    true_vel[i] = true_vel[i - 1] + true_acc[i] * system_dt;
    true_pos[i] = true_pos[i - 1] + true_vel[i] * system_dt;

    // New measurement comes once every M samples of the system
    if (i % M == 1)
      measured_pos[i] = true_pos[i] + measurement_noise(generator);
    else
      measured_pos[i] = measured_pos[i - 1];

    // Here we do the magic
    kf.updateState({believed_acc[i]}, {measured_pos[i]});

    estimated_pos[i] = kf.getEstimate()(0);
  }

  // Plot
  plt::title("Estimate of position");
  plt::xlabel("Time [s]");
  plt::ylabel("Position [m]");
  plt::named_plot("Truth", time, true_pos, "--");
  plt::named_plot("Measure", time, measured_pos, "-");
  plt::named_plot("Estimate", time, estimated_pos, "-");
  plt::legend();
  plt::grid(true);
  plt::fill_between(time, vector<double>(N, 0), true_pos, {{string("visible"), string("true")}, {string("alpha"), string("0.4")}});
  plt::xlim(0.0, simulation_time - system_dt);
  plt::save("./kf_result.png");
  plt::show();

  return 0;
}
