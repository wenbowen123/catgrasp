/*
Ball Query with BatchIdx & Clustering Algorithm
Written by Li Jiang
All Rights Reserved 2020.
*/

#include "bfs_cluster.h"
#include "nanoflann.hpp"
#include "eigen3/Eigen/Dense"

using KdtreeEigen = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>;


/* ================================== ballquery_batch_p ================================== */
// input xyz: (n, 3) float
// input batch_idxs: (n) int
// input batch_offsets: (B+1) int, batch_offsets[-1]
// output idx: (n * meanActive) dim 0 for number of points in the ball, idx in n
// output start_len: (n, 2), int, initially all zero
int ballquery_batch_p(at::Tensor xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor batch_offsets_tensor, at::Tensor idx_tensor, at::Tensor start_len_tensor, int n, int meanActive, float radius){
    const float *xyz = xyz_tensor.data<float>();
    const int *batch_idxs = batch_idxs_tensor.data<int>();
    const int *batch_offsets = batch_offsets_tensor.data<int>();
    int *idx = idx_tensor.data<int>();
    int *start_len = start_len_tensor.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int cumsum = ballquery_batch_p_cuda(n, meanActive, radius, xyz, batch_idxs, batch_offsets, idx, start_len, stream);
    return cumsum;
}

/* ================================== bfs_cluster ================================== */
ConnectedComponent find_cc(Int idx, int *semantic_label, Int *ball_query_idxs, int *start_len, int *visited){
    ConnectedComponent cc;
    cc.addPoint(idx);
    visited[idx] = 1;

    std::queue<Int> Q;
    assert(Q.empty());
    Q.push(idx);

    while(!Q.empty()){
        Int cur = Q.front(); Q.pop();
        int start = start_len[cur * 2];
        int len = start_len[cur * 2 + 1];
        int label_cur = semantic_label[cur];
        for(Int i = start; i < start + len; i++){
            Int idx_i = ball_query_idxs[i];
            if(semantic_label[idx_i] != label_cur) continue;
            if(visited[idx_i] == 1) continue;

            cc.addPoint(idx_i);
            visited[idx_i] = 1;

            Q.push(idx_i);
        }
    }
    return cc;
}

//input: semantic_label, int, N
//input: ball_query_idxs, Int, (nActive)
//input: start_len, int, (N, 2)
//output: clusters, CCs
int get_clusters(int *semantic_label, Int *ball_query_idxs, int *start_len, const Int nPoint, int threshold, ConnectedComponents &clusters){
    int visited[nPoint] = {0};

    int sumNPoint = 0;
    for(Int i = 0; i < nPoint; i++){
        if(visited[i] == 0){
            ConnectedComponent CC = find_cc(i, semantic_label, ball_query_idxs, start_len, visited);
            if((int)CC.pt_idxs.size() >= threshold){
                clusters.push_back(CC);
                sumNPoint += (int)CC.pt_idxs.size();
            }
        }
    }

    return sumNPoint;
}

void fill_cluster_idxs_(ConnectedComponents &CCs, int *cluster_idxs, int *cluster_offsets){
    for(int i = 0; i < (int)CCs.size(); i++){
        cluster_offsets[i + 1] = cluster_offsets[i] + (int)CCs[i].pt_idxs.size();
        for(int j = 0; j < (int)CCs[i].pt_idxs.size(); j++){
            int idx = CCs[i].pt_idxs[j];
            cluster_idxs[(cluster_offsets[i] + j) * 2 + 0] = i;
            cluster_idxs[(cluster_offsets[i] + j) * 2 + 1] = idx;
        }
    }
}

//input: semantic_label, int, N
//input: ball_query_idxs_tensor, int, (nActive)
//input: start_len, int, (N, 2)
//output: cluster_idxs, int (sumNPoint, 2), dim 1 First for cluster_id, second for corresponding point idxs in N
//output: cluster_offsets, int (nCluster + 1), each cluster's start pos
void bfs_cluster(at::Tensor semantic_label_tensor, at::Tensor ball_query_idxs_tensor, at::Tensor start_len_tensor, at::Tensor cluster_idxs_tensor, at::Tensor cluster_offsets_tensor, const int N, int threshold){
    int *semantic_label = semantic_label_tensor.data<int>();
    Int *ball_query_idxs = ball_query_idxs_tensor.data<Int>();
    int *start_len = start_len_tensor.data<int>();

    ConnectedComponents CCs;
    int sumNPoint = get_clusters(semantic_label, ball_query_idxs, start_len, N, threshold, CCs);

    int nCluster = (int)CCs.size();
    cluster_idxs_tensor.resize_({sumNPoint, 2});
    cluster_offsets_tensor.resize_({nCluster + 1});
    cluster_idxs_tensor.zero_();
    cluster_offsets_tensor.zero_();

    int *cluster_idxs = cluster_idxs_tensor.data<int>();
    int *cluster_offsets = cluster_offsets_tensor.data<int>();

    fill_cluster_idxs_(CCs, cluster_idxs, cluster_offsets);
}



ConnectedComponent find_cc_kdtree(Int idx, KdtreeEigen &kdtree, const float search_radius, const Eigen::MatrixXf &xyz, const std::vector<int> &semantic_label, int *visited){
    ConnectedComponent cc;
    cc.addPoint(idx);
    visited[idx] = 1;

    std::queue<Int> Q;
    assert(Q.empty());
    Q.push(idx);

    while(!Q.empty()){
        Int cur = Q.front();
        Q.pop();
        int label_cur = semantic_label[cur];
        for(Int i = 0; i < xyz.rows(); i++){
            Eigen::MatrixXf query_pt = xyz.block(i,0,1,3);
            std::vector<std::pair<long int, float> > indices_dists;
            nanoflann::SearchParams params;
            auto nMatches = kdtree.index->radiusSearch(query_pt.data(), search_radius, indices_dists, params);
            for (int j=0;j<indices_dists.size();j++)
            {
                auto idx_i = indices_dists[j].first;
                if(semantic_label[idx_i] != label_cur) continue;
                if(visited[idx_i] == 1) continue;

                cc.addPoint(idx_i);
                visited[idx_i] = 1;

                Q.push(idx_i);
            }
        }
    }
    return cc;
}



/**
 * @brief Get the clusters kdtree object, Process one batch
 *
 * @param xyz
 * @param kdtree
 * @param semantic_label
 * @param ball_query_idxs
 * @param start_len
 * @param nPoint
 * @param threshold
 * @param clusters
 * @return int
 */
int get_clusters_kdtree(const Eigen::MatrixXf &xyz, KdtreeEigen &kdtree, const float search_radius, const std::vector<int> &semantic_label, int threshold, ConnectedComponents &clusters){
    int visited[xyz.rows()] = {0};

    int sumNPoint = 0;
    for(Int i = 0; i < xyz.rows(); i++){
        if(visited[i] == 0){
            ConnectedComponent CC = find_cc_kdtree(i, kdtree, search_radius, xyz, semantic_label, visited);
            if((int)CC.pt_idxs.size() >= threshold){
                clusters.push_back(CC);
                sumNPoint += (int)CC.pt_idxs.size();
            }
        }
    }

    return sumNPoint;
}



/**
 * @brief Use kdtree to find neighbors online, this ensures completeness and saves memory
 *
 * @param xyz_tensor: (N,3)
 * @param semantic_label_tensor
 * @param batch_offsets_tensor (1+B) start pos (in N) of each batch
 * @param N
 * @param threshold: max number of points inside a cluster\
 * ------------------------
 * @cluster_idxs: int (sumNPoint, 2), dim 1 First for cluster_id, second for corresponding point idxs in N
 * @cluster_offsets: int (nCluster + 1), each cluster's start pos
 */
void bfs_cluster_kdtree(at::Tensor xyz_tensor, at::Tensor semantic_label_tensor, at::Tensor batch_offsets_tensor, at::Tensor cluster_idxs_tensor, at::Tensor cluster_offsets_tensor, const int N, const int threshold, const float search_radius)
{
    float *xyz = xyz_tensor.reshape({N*3}).data<float>();
    int *semantic_label = semantic_label_tensor.data<int>();
    int *batch_offsets = batch_offsets_tensor.data<int>();
    const int B = batch_offsets_tensor.size(0)-1;

    Eigen::MatrixXf xyz_mat(N*3,1);
    for (int i=0;i<N*3;i++)
    {
        xyz_mat(i,0) = xyz[i];
    }
    xyz_mat.resize(N,3);


    int sumNPoint = 0;
    ConnectedComponents CCs;
    for (int b=0;b<B;b++)
    {
        printf("bfs_cluster_kdtree batch %d/%d\n",b,B);
        Eigen::MatrixXf cur_batch_xyz = xyz_mat.block(batch_offsets[b],0,batch_offsets[b+1]-batch_offsets[b],3);
        KdtreeEigen kdtree(3,std::cref(cur_batch_xyz),10);
        std::vector<int> cur_batch_semantic_label(semantic_label+batch_offsets[b], semantic_label+batch_offsets[b+1]);
        ConnectedComponents cur_batch_CCs;
        int cur_batch_sumNPoint = get_clusters_kdtree(cur_batch_xyz, kdtree, search_radius, cur_batch_semantic_label, threshold, cur_batch_CCs);
        sumNPoint += cur_batch_sumNPoint;
        for (size_t i=0;i<cur_batch_CCs.size();i++)
        {
            auto &pt_idxs = cur_batch_CCs[i].pt_idxs;
            for (size_t j=0;j<pt_idxs.size();j++)
            {
                pt_idxs[j] += batch_offsets[b];
            }
            CCs.push_back(cur_batch_CCs[i]);
        }
    }

    int nCluster = (int)CCs.size();
    cluster_idxs_tensor.resize_({sumNPoint, 2});
    cluster_offsets_tensor.resize_({nCluster + 1});
    cluster_idxs_tensor.zero_();
    cluster_offsets_tensor.zero_();

    int *cluster_idxs = cluster_idxs_tensor.data<int>();
    int *cluster_offsets = cluster_offsets_tensor.data<int>();

    fill_cluster_idxs_(CCs, cluster_idxs, cluster_offsets);
}