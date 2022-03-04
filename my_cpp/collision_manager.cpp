#include "collision_manager.h"


CollisionManager::CollisionManager()
{

}


CollisionManager::~CollisionManager()
{

}

int CollisionManager::registerMesh(Eigen::Ref<const Eigen::MatrixXf> V, Eigen::Ref<const Eigen::MatrixXi> F)
{
  if (V.cols()!=3)
  {
    printf("vertices shape wrong: %dx%d\n",V.rows(),V.cols());
    exit(1);
  }
  if (F.cols()!=3)
  {
    printf("faces shape wrong: %dx%d\n",F.rows(),F.cols());
    exit(1);
  }

  std::vector<Vector3f> vertices;
  std::vector<Triangle> triangles;

  for (int i=0;i<V.rows();i++)
  {
    vertices.push_back(Vector3f(V(i,0),V(i,1),V(i,2)));
  }

  for (int i=0;i<F.rows();i++)
  {
    triangles.push_back(Triangle(F(i,0),F(i,1),F(i,2)));
  }

  std::shared_ptr<Model> geom = std::make_shared<Model>();
  geom->beginModel();
  geom->addSubModel(vertices, triangles);
  geom->endModel();
  CollisionObject<float> obj = CollisionObject<float>(geom);

  int ob_id = _obs.size();
  _obs.push_back(obj);

  return ob_id;

}


int CollisionManager::registerPointCloud(Eigen::Ref<const Eigen::MatrixXf> pts, const float resolution)
{
  if (pts.cols()!=3)
  {
    printf("point cloud shape wrong: %dx%d\n",pts.rows(),pts.cols());
    exit(1);
  }

  std::shared_ptr<octomap::OcTree> octree(new octomap::OcTree(resolution));
  for (int i=0;i<pts.rows();i++)
  {
    octree->updateNode(octomap::point3d(pts(i,0),pts(i,1),pts(i,2)), true);
  }

  OcTree<float>* tree = new OcTree<float>(octree);
  CollisionObject<float> tree_obj((std::shared_ptr<CollisionGeometry<float>>(tree)));

  int ob_id = _obs.size();
  _obs.push_back(tree_obj);

  return ob_id;

}



void CollisionManager::setTransform(Eigen::Ref<const Eigen::MatrixXf> pose, const int ob_id)
{
  if (pose.rows()!=4 || pose.cols()!=4)
  {
    printf("pose shape wrong: %dx%d\n",pose.rows(),pose.cols());
    exit(1);
  }

  _obs[ob_id].setTransform(pose.block(0,0,3,3),pose.block(0,3,3,1));
}


bool CollisionManager::isAnyCollision()
{
  for (int i=0;i<_obs.size();i++)
  {
    for (int j=i+1;j<_obs.size();j++)
    {
      const auto &oi = _obs[i];
      const auto &oj = _obs[j];
      CollisionRequest<float> request;
      CollisionResult<float> result;
      collide(&oi, &oj, request, result);
      if (result.isCollision())
      {
        return true;
      }
    }
  }
  return false;
}


