/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 30;
  std::default_random_engine gen;

  particles.resize(num_particles);
  weights.resize(num_particles);

  auto std_x = std[0];
  auto std_y = std[1];
  auto std_theta = std[2];

  for (uint i = 0; i < num_particles; i++)
  {
    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_theta);

    Particle particle;
    particle.id = i;
    particle.weight = 1.0;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particles[i] = particle;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;

  auto x_std = std_pos[0];
  auto y_std = std_pos[1];
  auto yaw_std = std_pos[2];

  double new_x, new_y, new_theta;

  for (uint i = 0; i < num_particles; i++)
  {
    if (yaw_rate == 0)
    {
      new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      new_theta = particles[i].theta;
    }
    else
    {
      new_x = particles[i].x +
              (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      new_y = particles[i].y +
              (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      new_theta = particles[i].theta + yaw_rate * delta_t;
    }
    std::normal_distribution<double> dist_x(new_x, x_std);
    particles[i].x = dist_x(gen);
    std::normal_distribution<double> dist_y(new_y, y_std);
    particles[i].y = dist_y(gen);
    std::normal_distribution<double> dist_theta(new_theta, yaw_std);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  auto distance_square = [](const LandmarkObs &observation_1, const LandmarkObs &observation_2) {
    return pow((observation_1.x - observation_2.x), 2) + pow((observation_1.y - observation_2.y), 2);
  };

  for (auto &observation_landmark : observations)
  {
    int closest_id = 0;
    auto min_distance_square = distance_square(observation_landmark, predicted[0]);

    for (uint i = 1; i < predicted.size(); i++)
    {
      auto distance = distance_square(observation_landmark, predicted[i]);
      if (distance < min_distance_square)
      {
        closest_id = i;
        min_distance_square = distance;
      }
    }
    observation_landmark.id = closest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  const auto gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  const auto gauss_den_x = 2 * pow(std_landmark[0], 2);
  const auto gauss_den_y = 2 * pow(std_landmark[1], 2);

  for (uint i = 0; i < particles.size(); ++i)
  {
    auto weight = 1.0;

    vector<LandmarkObs> landmarks_in_sensor_range;

    // get the nearby landmarks
    for (uint j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      // select landmarks in sensor range
      if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) < sensor_range)
      {
        landmarks_in_sensor_range.push_back({map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
      }
    }

    for (uint j = 0; j < observations.size(); j++)
    {
      auto cos_theta = cos(particles[i].theta);
      auto sin_theta = sin(particles[i].theta);
      // transform observation position from vehicle to map coordinate
      auto x_map = particles[i].x + cos_theta * observations[j].x - sin_theta * observations[j].y;
      auto y_map = particles[i].y + sin_theta * observations[j].x + cos_theta * observations[j].y;

      // get the nearest landmark
      auto nearest_landmark = std::min_element(landmarks_in_sensor_range.begin(), landmarks_in_sensor_range.end(),
                                               [=](const LandmarkObs &landmark1, const LandmarkObs &landmark2) {
                                                 return dist(x_map, y_map, landmark1.x, landmark1.y) <
                                                        dist(x_map, y_map, landmark2.x, landmark2.y);
                                               });

      //Multivariate-Gaussian probability
      auto exponent = pow(x_map - nearest_landmark->x, 2) / gauss_den_x + pow(y_map - nearest_landmark->y, 2) / gauss_den_y;
      weight *= gauss_norm * exp(-exponent);
    }
    particles[i].weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<Particle> new_particles(num_particles);

  double beta = 0;
  int index = rand() % num_particles;
  auto max_weight_element = max_element(weights.begin(), weights.end());

  for (uint i = 0; i < num_particles; ++i)
  {

    beta += (rand() / (RAND_MAX + 1.0)) * (2 * (*max_weight_element));
    while (weights[index] < beta)
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles[i] = particles[index];
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}