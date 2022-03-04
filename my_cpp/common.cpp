#include "common.h"
#include "collision_manager.h"

#ifdef IKFAST_NAMESPACE
using namespace IKFAST_NAMESPACE;
#endif


std::vector<std::vector<double>> get_ik_within_limits(const Eigen::Matrix4f &ee_in_base, const std::vector<double> &upper, const std::vector<double> &lower)
{

  using namespace ikfast;
  IkSolutionList<double> solutions;

  std::vector<double> vfree(GetNumFreeParameters());
  double eerot[9], eetrans[3];

  for (std::size_t i = 0; i < 3; ++i)
  {
    eetrans[i] = ee_in_base(i,3);
  }
  for (int h=0;h<3;h++)
  {
    for (int w=0;w<3;w++)
    {
      eerot[h*3+w] = ee_in_base(h,w);
    }
  }

  bool b_success = ComputeIk(eetrans, eerot, &vfree[0], solutions);

  std::vector<std::vector<double> > solution_list;
  if (!b_success)
  {
      return solution_list;
  }

  std::vector<double> solvalues(GetNumJoints());

  for (std::size_t i = 0; i < solutions.GetNumSolutions(); ++i)
  {
      const IkSolutionBase<double> &sol = solutions.GetSolution(i);
      std::vector<double> vsolfree(sol.GetFree().size());
      sol.GetSolution(&solvalues[0],
                      vsolfree.size() > 0 ? &vsolfree[0] : NULL);

      std::vector<double> individual_solution = std::vector<double>(GetNumJoints());
      for (std::size_t j = 0; j < solvalues.size(); ++j)
      {
          individual_solution[j] = solvalues[j];
      }

      bool isbad = false;
      for (int ii=0;ii<individual_solution.size();ii++)
      {
        if (individual_solution[ii]>upper[ii])
        {
          isbad = true;
          break;
        }
        if (individual_solution[ii]<lower[ii])
        {
          isbad = true;
          break;
        }
      }
      if (isbad) continue;

      solution_list.push_back(individual_solution);
  }
  return solution_list;
}


Eigen::Matrix3f directionVecToRotation(Eigen::Vector3f direction, const Eigen::Vector3f &ref)
{
  direction.normalize();
  Eigen::Vector3f v = direction.cross(ref);
  if ((v-Eigen::Vector3f::Zero()).norm()<1e-5)
  {
    return Eigen::Matrix3f::Identity();
  }
  float s = v.norm();
  float c = direction.dot(ref);
  Eigen::Matrix3f v_skew;
  v_skew<<0,-v(2),v(1),
          v(2),0,-v(0),
          -v(1),v(0),0;

  Eigen::Matrix3f R(Eigen::Matrix3f::Identity());

  if (s==0)
  {
    R<<1,0,0,
      0,-1,0,
      0,0,-1;
  }
  else
  {
    R = Eigen::Matrix3f::Identity()+v_skew+v_skew*v_skew*(1-c)/(s*s);
    R = R.transpose().eval();
  }

  Eigen::JacobiSVD<Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  R = svd.matrixU()*svd.matrixV().transpose();
  if ((R*ref-direction).norm()>1e-3)
  {
    printf("In directionVecToRotMat, rotation error %f\n",(R*ref-direction).norm());
    std::cout<<"R:\n"<<R<<std::endl;
    std::cout<<"direction: "<<direction.transpose()<<std::endl;
  }
  return R;
}


vectorMatrix4f augmentGraspPoses(const Eigen::Matrix3f &R0, const Eigen::Vector3f &selected_point, const Eigen::MatrixXf &sphere_pts, float inplane_rot_step, float hand_depth, float approach_step, float init_bite)
{
  std::vector<Eigen::Matrix3f> Rs = {R0};
  for (int i=0;i<sphere_pts.size();i++)
  {
    Eigen::Vector3f sphere_pt = sphere_pts.row(i);
    Eigen::Matrix3f R_sphere = directionVecToRotation(sphere_pt,Eigen::Vector3f(1,0,0));
    for (float x_rot=0;x_rot<180;x_rot+=inplane_rot_step)
    {
      Eigen::AngleAxisf rollAngle(x_rot/180.0*M_PI, Eigen::Vector3f::UnitX());
      Eigen::AngleAxisf pitchAngle(0, Eigen::Vector3f::UnitY());
      Eigen::AngleAxisf yawAngle(0, Eigen::Vector3f::UnitZ());
      Eigen::Quaternionf q = yawAngle * pitchAngle * rollAngle;
      Eigen::Matrix3f R_inplane = q.normalized().matrix();
      Rs.push_back(R0*R_sphere*R_inplane);
    }
  }

  vectorMatrix4f grasp_poses;
  grasp_poses.reserve(Rs.size());
  for (int i=0;i<Rs.size();i++)
  {
    Eigen::Matrix3f R = Rs[i];
    Eigen::JacobiSVD<Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU()*svd.matrixV().transpose();
    Eigen::Vector3f approach_dir = R.block(0,0,3,1);
    for (float d=0;d<hand_depth;d+=approach_step)
    {
      Eigen::Vector3f grasp_center = selected_point+init_bite*approach_dir+approach_dir*d;
      Eigen::Matrix4f grasp_pose(Eigen::Matrix4f::Identity());
      grasp_pose.block(0,0,3,3) = R;
      grasp_pose.block(0,3,3,1) = grasp_center;
      grasp_poses.push_back(grasp_pose);
    }
  }

  return grasp_poses;
}


vectorMatrix4f filterGraspPose(const vectorMatrix4f grasp_poses, const vectorMatrix4f symmetry_tfs, const Eigen::Matrix4f nocs_pose, const Eigen::Matrix4f canonical_to_nocs_transform, const Eigen::Matrix4f cam_in_world, const Eigen::Matrix4f ee_in_grasp, const Eigen::Matrix4f gripper_in_grasp, bool filter_approach_dir_face_camera, bool filter_ik, bool adjust_collision_pose, const std::vector<double> upper, const std::vector<double> lower, const Eigen::MatrixXf gripper_vertices, const Eigen::MatrixXi gripper_faces, const Eigen::MatrixXf gripper_enclosed_vertices, const Eigen::MatrixXi gripper_enclosed_faces, const Eigen::MatrixXf gripper_collision_pts, const Eigen::MatrixXf gripper_enclosed_collision_pts, float octo_resolution, bool verbose)
{
  vectorMatrix4f out;
  Eigen::Matrix4f canonical_to_cam = nocs_pose*canonical_to_nocs_transform;
  std::cout<<"canonical_to_cam:\n"<<canonical_to_cam<<"\n\n";

  int n_approach_dir_rej = 0;
  int n_ik_rej = 0;
  int n_open_gripper_rej = 0;
  int n_close_gripper_rej = 0;

omp_set_num_threads(int(std::thread::hardware_concurrency()));
#pragma omp parallel firstprivate(grasp_poses,symmetry_tfs,nocs_pose,canonical_to_nocs_transform,cam_in_world,ee_in_grasp,gripper_in_grasp,upper,lower,gripper_vertices,gripper_faces,gripper_enclosed_vertices,gripper_enclosed_faces,gripper_collision_pts,gripper_enclosed_collision_pts,canonical_to_cam)
{
  vectorMatrix4f out_local;
  int n_approach_dir_rej_local = 0;
  int n_ik_rej_local = 0;
  int n_open_gripper_rej_local = 0;
  int n_close_gripper_rej_local = 0;

  CollisionManager cm;
  int gripper_id = cm.registerMesh(gripper_vertices,gripper_faces);
  cm.registerPointCloud(gripper_collision_pts,octo_resolution);

  CollisionManager cm_bg;
  int gripper_enclosed_id = cm_bg.registerMesh(gripper_enclosed_vertices,gripper_enclosed_faces);
  cm_bg.registerPointCloud(gripper_enclosed_collision_pts,octo_resolution);

  #pragma omp for schedule(dynamic)
  for (int i=0;i<grasp_poses.size();i++)
  {
    const auto &grasp_pose = grasp_poses[i];
    for (int j=0;j<symmetry_tfs.size();j++)
    {
      const auto &tf = symmetry_tfs[j];
      Eigen::Matrix4f tmp_grasp_pose = tf*grasp_pose;
      Eigen::Matrix4f grasp_in_cam = canonical_to_cam*tmp_grasp_pose;

      for (int col=0;col<3;col++)
      {
        grasp_in_cam.block(0,col,3,1).normalize();
      }

      if (filter_approach_dir_face_camera)
      {
        Eigen::Vector3f approach_dir = grasp_in_cam.block(0,0,3,1);
        approach_dir.normalize();
        float dot = approach_dir.dot(Eigen::Vector3f(0,0,1));
        if (dot<0)
        {
          if (verbose)
          {
            n_approach_dir_rej_local++;
          }
          continue;
        }
      }

      if (filter_ik)
      {
        Eigen::Matrix4f ee_in_base = cam_in_world*grasp_in_cam*ee_in_grasp;
        auto sols = get_ik_within_limits(ee_in_base,upper,lower);
        if (sols.size()==0)
        {
          if (verbose)
          {
            n_ik_rej_local++;
          }
          continue;
        }
      }

      if (!adjust_collision_pose)
      {
        Eigen::Matrix4f gripper_in_cam = grasp_in_cam*gripper_in_grasp;
        cm.setTransform(gripper_in_cam,gripper_id);
        if (cm.isAnyCollision())
        {
          if (verbose)
          {
            n_open_gripper_rej_local++;
          }
          continue;
        }

        cm_bg.setTransform(gripper_in_cam,gripper_enclosed_id);
        if (cm_bg.isAnyCollision())
        {
          if (verbose)
          {
            n_close_gripper_rej_local++;
          }
          continue;
        }
      }
      else
      {
        Eigen::Vector3f major_dir = grasp_in_cam.block(0,1,3,1);
        bool found = false;
        for (float step=0.0;step<=0.003;step+=0.001)
        {
          std::vector<int> signs = {1,-1};
          if (step==0)
          {
            signs = {1};
          }
          for (auto sign:signs)
          {
            Eigen::Matrix4f cur_grasp_in_cam = grasp_in_cam;
            cur_grasp_in_cam.block(0,3,3,1) += step*major_dir*sign;
            Eigen::Matrix4f cur_gripper_in_cam = cur_grasp_in_cam*gripper_in_grasp;
            cm.setTransform(cur_gripper_in_cam,gripper_id);
            if (cm.isAnyCollision())
            {
              continue;
            }

            cm_bg.setTransform(cur_gripper_in_cam,gripper_enclosed_id);
            if (cm_bg.isAnyCollision())
            {
              continue;
            }

            grasp_in_cam = cur_grasp_in_cam;
            found = true;
            break;
          }
          if (found)
          {
            break;
          }
        }

        if (!found)
        {
          grasp_in_cam.setZero();
          n_open_gripper_rej_local++;
        }
      }

      if (grasp_in_cam!=Eigen::Matrix4f::Zero())
      {
        out_local.push_back(grasp_in_cam);
      }
    }
  }

  #pragma omp critical
  {
    n_approach_dir_rej += n_approach_dir_rej_local;
    n_ik_rej += n_ik_rej_local;
    n_open_gripper_rej += n_open_gripper_rej_local;
    n_close_gripper_rej += n_close_gripper_rej_local;
    for (int i=0;i<out_local.size();i++)
    {
      out.push_back(out_local[i]);
    }
  }
}

  if (verbose)
  {
    printf("n_approach_dir_rej=%d, n_ik_rej=%d, n_open_gripper_rej=%d, n_close_gripper_rej=%d\n",n_approach_dir_rej,n_ik_rej,n_open_gripper_rej,n_close_gripper_rej);
  }
  return out;
}


Eigen::MatrixXf makeOccupancyGridFromCloudScan(const Eigen::MatrixXf &pts, const Eigen::Matrix3f &K, float resolution)
{
  using namespace octomap;
  using namespace octomath;

  assert(pts.cols()==3);

  octomap::OcTree tree (resolution);

  Pointcloud p;
  int umin = 999999;
  int umax = -999999;
  int vmin = 999999;
  int vmax = -999999;
  for (int i=0;i<pts.rows();i++)
  {
    point3d pt(pts(i,0),pts(i,1),pts(i,2));
    p.push_back(pt);
    Eigen::Vector3f projected = K*pts.row(i).transpose();
    int u = std::round(projected(0)/projected(2));
    int v = std::round(projected(1)/projected(2));
    umin = std::min(umin,u);
    vmin = std::min(vmin,v);
    umax = std::max(umax,u);
    vmax = std::max(vmax,v);
  }
  point3d sensor_origin(0.0f, 0.0f, 0.0f);
  tree.insertPointCloud(p, sensor_origin);

  Eigen::Vector3f max_xyz = pts.colwise().maxCoeff();
  Eigen::Vector3f min_xyz = pts.colwise().minCoeff();
  float xmax = max_xyz(0);
  float ymax = max_xyz(1);
  float zmax = max_xyz(2);
  float xmin = min_xyz(0);
  float ymin = min_xyz(1);
  float zmin = min_xyz(2);

  printf("xyz max=(%f,%f,%f), min=(%f,%f,%f), dimension=(%f,%f,%f)\n",xmax,ymax,zmax,xmin,ymin,zmin,xmax-xmin,ymax-ymin,zmax-zmin);


  std::vector<Eigen::Vector3f> occupied_pts;
  occupied_pts.reserve( int((xmax-xmin)/resolution * (ymax-ymin)/resolution * (zmax-zmin)/resolution) );

omp_set_num_threads(int(std::thread::hardware_concurrency()));
#pragma omp parallel firstprivate(xmin,xmax,ymin,ymax,zmin,zmax,resolution,sensor_origin,tree)
{
  float pad = 0.005;
  std::vector<Eigen::Vector3f> occupied_pts_local;

  int max_xi = (xmax+pad-(xmin-pad))/resolution;
  int max_yi = (ymax+pad-(ymin-pad))/resolution;
  int max_zi = (zmax+pad-(zmin-pad))/resolution;
  float max_range = std::sqrt(std::pow(xmax+pad,2) + std::pow(ymax+pad,2) + std::pow(zmax+pad,2));

  #pragma omp for schedule(dynamic)
  for (int xi=0;xi<max_xi;xi++)
  {
    for (int yi=0;yi<max_yi;yi++)
    {
      for (int zi=0;zi<max_zi;zi++)
      {
        float x = xmin-pad+xi*resolution;
        float y = ymin-pad+yi*resolution;
        float z = zmin-pad+zi*resolution;
        Eigen::Vector3f dir(x,y,z);
        dir.normalize();
        point3d dir_octo(dir(0),dir(1),dir(2));
        point3d end_pt(0,0,0);
        bool is_hit = tree.castRay(sensor_origin,dir_octo,end_pt,true,max_range);
        bool is_empty = true;
        if (is_hit)
        {
          float dist_query = std::sqrt(x*x+y*y+z*z);
          float dist = end_pt.norm();
          if (dist<=dist_query)
          {
            is_empty = false;
          }
        }

        if (!is_empty)
        {
          occupied_pts_local.push_back(Eigen::Vector3f(x,y,z));
        }
      }
    }
  }

  #pragma omp critical
  {
    for (int i=0;i<occupied_pts_local.size();i++)
    {
      occupied_pts.push_back(occupied_pts_local[i]);
    }
  }
}


  Eigen::MatrixXf out = Eigen::MatrixXf::Zero(occupied_pts.size(),3);
  for (int i=0;i<occupied_pts.size();i++)
  {
    out.row(i) = occupied_pts[i].transpose();
  }

  return out;

}
