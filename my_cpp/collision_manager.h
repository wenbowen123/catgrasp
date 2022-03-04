#ifndef COLLISION_MANAGER_HH
#define COLLISION_MANAGER_HH

#include <vector>
#include <Eigen/Dense>
// STL
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
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
#include "fcl/config.h"
#include "fcl/geometry/octree/octree.h"
#include "fcl/narrowphase/collision.h"
#include "fcl/broadphase/broadphase_bruteforce.h"
#include "fcl/broadphase/broadphase_spatialhash.h"
#include "fcl/broadphase/broadphase_SaP.h"
#include "fcl/broadphase/broadphase_SSaP.h"
#include "fcl/broadphase/broadphase_interval_tree.h"
#include "fcl/broadphase/broadphase_dynamic_AABB_tree.h"
#include "fcl/broadphase/broadphase_dynamic_AABB_tree_array.h"
#include "fcl/broadphase/default_broadphase_callbacks.h"
#include "fcl/geometry/geometric_shape_to_BVH_model.h"
#include <eigen3/Eigen/Dense>
#include "pybind11/eigen.h"


using uchar = unsigned char;
using namespace fcl;
typedef BVHModel<OBBRSSf> Model;


class CollisionManager
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CollisionManager();
  ~CollisionManager();
  int registerMesh(Eigen::Ref<const Eigen::MatrixXf> V, Eigen::Ref<const Eigen::MatrixXi> F);
  int registerPointCloud(Eigen::Ref<const Eigen::MatrixXf> pts, const float resolution);
  void setTransform(Eigen::Ref<const Eigen::MatrixXf> pose, const int ob_id);
  bool isAnyCollision();

public:
  std::vector<CollisionObject<float>> _obs;
};





#endif