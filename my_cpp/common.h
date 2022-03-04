#ifndef MYCPP_COMMON_HH
#define MYCPP_COMMON_HH

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <Eigen/Dense>
// STL
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <set>
#include <unordered_set>
#include <time.h>
#include <queue>
#include <climits>
#include <boost/assign.hpp>
#include <boost/algorithm/string.hpp>
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <string>
#include <limits.h>
#include <unistd.h>
#include <memory>
#include <math.h>
#include <boost/format.hpp>
#include <numeric>
#include <thread>
#include <omp.h>
#include <exception>
#include <deque>
#include <random>
#include <octomap/octomap.h>
#include <octomap/math/Utils.h>


#ifndef IKFAST_HAS_LIBRARY
#define IKFAST_HAS_LIBRARY
#endif

#include "kuka_iiwa14/ikfast.h"


#ifdef IKFAST_NAMESPACE
using namespace IKFAST_NAMESPACE;
#endif

using vectorMatrix4f = std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>>;
class CollisionManager;

std::vector<std::vector<double>> get_ik_within_limits(const Eigen::Matrix4f &ee_in_base, const std::vector<double> &upper, const std::vector<double> &lower);

Eigen::Matrix3f directionVecToRotation(Eigen::Vector3f direction, const Eigen::Vector3f &ref);

vectorMatrix4f augmentGraspPoses(const Eigen::Matrix3f &R0, const Eigen::Vector3f &selected_point, const Eigen::MatrixXf &sphere_pts, float inplane_rot_step, float hand_depth, float approach_step, float init_bite);

vectorMatrix4f filterGraspPose(const vectorMatrix4f grasp_poses, const vectorMatrix4f symmetry_tfs, const Eigen::Matrix4f nocs_pose, const Eigen::Matrix4f canonical_to_nocs_transform, const Eigen::Matrix4f cam_in_world, const Eigen::Matrix4f ee_in_grasp, const Eigen::Matrix4f gripper_in_grasp, bool filter_approach_dir_face_camera, bool filter_ik, bool adjust_collision_pose, const std::vector<double> upper, const std::vector<double> lower, const Eigen::MatrixXf gripper_vertices, const Eigen::MatrixXi gripper_faces, const Eigen::MatrixXf gripper_enclosed_vertices, const Eigen::MatrixXi gripper_enclosed_faces, const Eigen::MatrixXf gripper_collision_pts, const Eigen::MatrixXf gripper_enclosed_collision_pts, float octo_resolution, bool verbose);
Eigen::MatrixXf makeOccupancyGridFromCloudScan(const Eigen::MatrixXf &pts, const Eigen::Matrix3f &K, float resolution);

#endif