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
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  num_particles = 200;  // TODO: Set the number of particles
  particles.resize(num_particles); // resize the vector of particles to avoid segmentation fault
  // create normal distributions for x, y, and theta.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  default_random_engine gen; // initializing of the random engine, to use "gen" in the later steps.
  
  // use samples from normal distribution to initialize the particles
  for (int i=0; i<num_particles; i++)
  {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
    particles.push_back(particles[i]);
  }
  
  // update the initialization status to true
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  // create random Gaussian noise for x, y, and theta. 
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  for (int i=0; i<num_particles; i++)
  {
	// if yaw rate is very small or 0
    if ( fabs(yaw_rate) < 0.0001)
    {
	  // when yaw rate is 0, the equation is: xf = x0 + v * (dt) * (cos(theta_0))
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
	  // when yaw rate is 0, the equation is: yf = y0 + v * (dt) * (sin(theta_0))
      particles[i].y += velocity * delta_t * sin(particles[i].theta);    
    }
    else
    {
	  // when yaw rate is not 0, the equation is: xf = x0 + (v/theta_dot) * ( sin(theta_0+theta_dot*dt) - sin(theta_0) )
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
	  // when yaw rate is not 0, the equation is: yf = x0 + (v/theta_dot) * ( cos(theta_0) - cos(theta_0+theta_dot*dt) )
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      // when yaw rate is not 0, the equation is: theta_f = theta_0 + theta_dot * dt
	  particles[i].theta += yaw_rate * delta_t;
    }
	
	//update x, y, and theta by adding the random Gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
    
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  // initialize variables to avoid comparing unsigned int with signed int
  unsigned int obs_size = observations.size();
  unsigned int pred_size = predicted.size();
  
  // the code logic of this dataAssociation part is referenced from "darienmt" https://github.com/darienmt/CarND-Kidnapped-Vehicle-P3/blob/master/src/particle_filter.cpp
  /* first initialize the minimal distance to a big number
  *  then initialized the map ID
  *  calculate the assign distance and compare with the minimal distance
  *  if the assign distance is less than the minimal distance, the minimal distance will be associzted with the assign distance
  *  and the minimal distance will be updated 
  */
  for ( unsigned int i=0; i<obs_size; i++)
  {
    double min_dist = numeric_limits<double>::max(); // first initialize the minimal distant to a big number
    int map_id = -1;
    
    for (unsigned int j=0; j<pred_size; j++)
    {
	  // use the dist function to calculate distance, equation is: sqrt( (x_observation - x_prediction)^2 + (y_observation - y_prediction)^2 )
      double assign_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
	
      if (assign_dist < min_dist)
      {
        min_dist = assign_dist;
        map_id = predicted[j].id;
      }
    }
    
	// update the observation id with the map id 
    observations[i].id = map_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
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
   
  /* In order to incorporated velocity and yaw rate measurements into the filter,
  *  the particle weights need to be updated by the readings of landmarks.
  *  1. find the landmarks within sensor range, if there are landmarks within sensor range, 
        the landmarks will be assigned to predictions
	 2. because the observations are given in the vehicle's coordinate system, 
	    transformation is required for both rotation and translation
	 3. for each observation, it needs to associated to the landmark index.
	    use the dataAssociation function defined previously
	 4. update the weights of each particle using a muli-variate Gaussian distribution.
	 the logic of this function is adapted from "dkarunakaran" https://github.com/dkarunakaran/carnd-kidnapped-vehicle-term2-p3/blob/master/src/particle_filter.cpp
  */
  for (int i=0; i<num_particles; i++)
  {
	// initialize particle variables
    double par_x = particles[i].x;
    double par_y = particles[i].y;
    double par_theta = particles[i].theta;
    
    vector<LandmarkObs> predictions;
    
    for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++)
    {
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      
	  // find the landmarks within sensor range
      if (fabs(landmark_x - par_x) <= sensor_range && fabs(landmark_y - par_y) <= sensor_range)
      {
        predictions.push_back(LandmarkObs {landmark_id, landmark_x, landmark_y});
      }
    }
    
	// observation transformation using equation 3.33 from http://planning.cs.uiuc.edu/node99.html
    vector<LandmarkObs> transformed_os;
    for (unsigned int j=0; j<observations.size(); j++)
    {
      double x_t = observations[j].x * cos(par_theta) - observations[j].y * sin(par_theta) + par_x;
      double y_t = observations[j].x * sin(par_theta) + observations[j].y * cos(par_theta)+ par_y;
      transformed_os.push_back(LandmarkObs{ observations[j].id, x_t, y_t });
    }
    
	// data association for observation and landmark index
    dataAssociation(predictions, transformed_os);
    particles[i].weight = 1.0; // reset particle weight
    
    for (unsigned int j=0; j<transformed_os.size(); j++)
    {
      double obs_x, obs_y, mu_x, mu_y;
      obs_x = transformed_os[j].x;
      obs_y = transformed_os[j].y;
      int associated_prediction = transformed_os[j].id;
      for (unsigned int k=0; k<predictions.size(); k++)
      {
        if (predictions[k].id == associated_prediction)
        {
          mu_x = predictions[k].x;
          mu_y = predictions[k].y;
        }
      }
      
	  // calculating the particle's final weight, the multivariate gaussian equation is adapted from lesson 5.
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double obs_w = (1 / (2 * M_PI * sig_x * sig_y)) * exp( -(pow(mu_x - obs_x , 2) / (2 * pow(sig_x , 2)) + (pow(mu_y - obs_y , 2) / (2 * pow(sig_y , 2)) )));
      
	  // update particle weight
      particles[i].weight *= obs_w;
    }
  }
    
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // find the max weight as mentioned in the logic from lesson 4
  vector<double> weights;
  double max_weight = numeric_limits<double>::min();
  for (int i=0; i<num_particles; i++)
  {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > max_weight)
    {
      max_weight = particles[i].weight;
    }
  }
  
  // generate random integer from num_particles
  uniform_int_distribution<int> dist_int(0, num_particles-1);
  // generate random floating number from 0.0 to 1.0
  float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  
  /* resampling wheel logic is adapted from lesson 4,
   * while w[index] < beta:
   *        beta = beta - w[index]
   *		index = index +1
   * 
   * 	select p[index]
   */
  default_random_engine gen;
  int index = dist_int(gen);
  double beta = 0;
  vector<Particle> resampled_particles;
  for(int i=0; i<num_particles; i++)
  {
    beta += r * max_weight * 2; // beta is in range [0, 2*max_weight]
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }
  
  // update the particle weights to the resampled particle weights
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}