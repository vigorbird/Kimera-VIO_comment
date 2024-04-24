/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   Mesh3D.cpp
 * @brief  Represents a Mesh of polygons in 3D
 * @author Antoni Rosinol
 */

#include "kimera-vio/mesh/Mesh.h"

#include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/persistence.hpp>

#include "kimera-vio/utils/UtilsNumerical.h"

namespace VIO {

/**
 * param[in]: polygon_dimension number of vertices per polygon (triangle = 3).
 */
template <typename VertexPositionType>
Mesh<VertexPositionType>::Mesh(const size_t& polygon_dimension)
    : vertex_to_lmk_id_map_(),
      lmk_id_to_vertex_map_(),
      vertices_mesh_(0, 1, CV_32FC3),
      vertices_mesh_normal_(),
      normals_computed_(false),
      vertices_mesh_color_(0, 0, CV_8UC3, cv::viz::Color::blue()),
      polygons_mesh_(0, 1, CV_32SC1),
      adjacency_matrix_(1, 1, CV_8UC1, cv::Scalar(0u)),
      face_hashes_(),
      polygon_dimension_(polygon_dimension) {
  CHECK_GE(polygon_dimension, 3) << "A polygon must have more than 2"
                                    " vertices";
}

/* -------------------------------------------------------------------------- */
template <typename VertexPositionType>
Mesh<VertexPositionType>::Mesh(const Mesh<VertexPositionType>& rhs_mesh)
    : vertex_to_lmk_id_map_(rhs_mesh.vertex_to_lmk_id_map_),
      lmk_id_to_vertex_map_(rhs_mesh.lmk_id_to_vertex_map_),
      vertices_mesh_(rhs_mesh.vertices_mesh_.clone()),        // CLONING!
      vertices_mesh_normal_(rhs_mesh.vertices_mesh_normal_),  // COPYING!
      normals_computed_(rhs_mesh.normals_computed_),
      vertices_mesh_color_(rhs_mesh.vertices_mesh_color_.clone()),  // CLONING!
      polygons_mesh_(rhs_mesh.polygons_mesh_.clone()),              // CLONING!
      adjacency_matrix_(rhs_mesh.adjacency_matrix_.clone()),        // CLONING!
      face_hashes_(rhs_mesh.face_hashes_),
      polygon_dimension_(rhs_mesh.polygon_dimension_) {
  VLOG(2) << "You are calling the copy ctor for a mesh... Cloning data.";
}

/* -------------------------------------------------------------------------- */
template <typename VertexPositionType>
Mesh<VertexPositionType>& Mesh<VertexPositionType>::operator=(
    const Mesh<VertexPositionType>& rhs_mesh) {
  // Check for self-assignment.
  if (&rhs_mesh == this) return *this;
  CHECK_EQ(polygon_dimension_, rhs_mesh.polygon_dimension_)
      << "The Mesh that you are trying to copy has different dimensions"
      << " for the polygons!";
  // Deep copy internal data.
  lmk_id_to_vertex_map_ = rhs_mesh.lmk_id_to_vertex_map_;
  vertex_to_lmk_id_map_ = rhs_mesh.vertex_to_lmk_id_map_;
  vertices_mesh_ = rhs_mesh.vertices_mesh_.clone();
  vertices_mesh_normal_ = rhs_mesh.vertices_mesh_normal_;
  normals_computed_ = rhs_mesh.normals_computed_;
  vertices_mesh_color_ = rhs_mesh.vertices_mesh_color_.clone();
  polygons_mesh_ = rhs_mesh.polygons_mesh_.clone();
  adjacency_matrix_ = rhs_mesh.adjacency_matrix_.clone();
  face_hashes_ = rhs_mesh.face_hashes_;
  return *this;
}

/* -------------------------------------------------------------------------- */
template <typename VertexPositionType>
void Mesh<VertexPositionType>::addPolygonToMesh(const Polygon& polygon) {
  // Update mesh connectivity
  CHECK_EQ(polygon.size(), polygon_dimension_)
      << "Trying to insert a polygon of different dimension than "
      << "the mesh's polygons.\n"
      << "Polygon dimension: " << polygon.size() << "\n"
      << "Mesh expected polygon dimension: " << polygon_dimension_ << ".\n";
  // Reset flag to know if normals are valid or not.
  normals_computed_ = false;

  // Update vertices in the mesh (this happens all the time, even if we
  // do not add a new triangle connectivity-wise).
  VertexIds vtx_ids;//这个三角形对应的全局mesh 顶点序号
  bool triangle_maybe_already_in_mesh = true;

  //只要有一个顶点不在全局的mesh中则认定triangle_maybe_already_in_mesh = false;
  //输入的多边形 = 一个三角形数据结构
  //1.遍历这个多边形的所有顶点 判断是否顶点在全局mesh中
  for (const VertexType& vertex : polygon) {
    const LandmarkId& lmk_id = vertex.getLmkId();
    VertexId existing_vtx_id;
    if (!getVtxIdForLmkId(lmk_id, &existing_vtx_id)) {
      // Vtx is not in the mesh, so no way the triangle is in the mesh.
      triangle_maybe_already_in_mesh = false;
    }
    //下面这个函数不是很重要的小函数
    const VertexId& vtx_id =
        updateMeshDataStructures(lmk_id,
                                 vertex.getVertexPosition(),
                                 vertex.getVertexColor(),
                                 vertex.getVertexNormal(),
                                 &vertex_to_lmk_id_map_,//输出，顶点到landmark索引
                                 &lmk_id_to_vertex_map_,//输出，landmark到顶点的索引
                                 &vertices_mesh_,//输出，mesh所有顶点
                                 &vertices_mesh_normal_,//输出，mesh所有顶点的法向量
                                 &vertices_mesh_color_);//输出，mesh所有顶点对应的颜色
    if (triangle_maybe_already_in_mesh) {
      // Just a small sanity check.
      CHECK_EQ(vtx_id, existing_vtx_id);
    }

    CHECK_NE(vtx_id, -1);
    vtx_ids.push_back(vtx_id);
  }//遍历所有的多边形顶点结束
  CHECK_EQ(vtx_ids.size(), polygon_dimension_);

  // DO NOT sort vtx_ids, or the normals will flip!
  //2.判断face是否在全局mesh中
  VertexIds sorted_vtx_ids = vtx_ids;
  std::sort(sorted_vtx_ids.begin(), sorted_vtx_ids.end());
  //将三个int得到一个唯一hash索引
  const auto& face_hash = UtilsNumerical::hashTriplet(sorted_vtx_ids[0], sorted_vtx_ids[1], sorted_vtx_ids[2]);
  //std::unordered_map<size_t, bool> face_hashes_;
  const auto& it = face_hashes_.find(face_hash);

  // Check the triangle is not already in the mesh
  CHECK_EQ(polygon_dimension_, 3) << "This doesn't work with non-triangles";
  bool triangle_in_mesh = false;
  //顶点在全局mesh中但是face不一定在
  if (triangle_maybe_already_in_mesh) {
    // Check that the triangle is not already in the mesh!
    if (it != face_hashes_.end()) {
      // LOG(ERROR) << "Found existing face with hash: " << face_hash;
      // Triangle already exists!
      triangle_in_mesh = true;
      CHECK(it->second);
      CHECK_EQ(adjacency_matrix_.at<uint8_t>(sorted_vtx_ids[0], sorted_vtx_ids[1]), 1u);
      CHECK_EQ(adjacency_matrix_.at<uint8_t>(sorted_vtx_ids[1], sorted_vtx_ids[2]), 1u);
      CHECK_EQ(adjacency_matrix_.at<uint8_t>(sorted_vtx_ids[2], sorted_vtx_ids[0]), 1u);
    } else {
      triangle_in_mesh = false;//face不在全局mesh里面
    }
  }

  //3.更新全局mesh的adjacency_matrix_ 和face信息
  if (!triangle_in_mesh) {//如果face不在全局mesh里面，就需要让这个新的face更新全局mesh的face信息
    // LOG(ERROR) << "Adding face with hash: " << face_hash;
    CHECK(it == face_hashes_.end()) << "Hash collision? This can happen but "
                                       "weird... Check your hashing function.";
    face_hashes_[face_hash] = true;

    // Update polygons_mesh_
    // Specify number of point ids per face in the mesh.
    //先更新全局mesh的全局face信息
    polygons_mesh_.push_back(static_cast<int>(polygon_dimension_));
    for (const VertexId& vtx_id : vtx_ids) {
      polygons_mesh_.push_back(static_cast<int>(vtx_id));
    }

    // Update adjacency matrix

    if (!triangle_maybe_already_in_mesh) {//顶点不在全局mesh中 需要对adjacency_matrix_进行扩容
      // There are new vertices!
      // TODO(Toni): this assumes that we never remove vertices!!
      // Add a new col/row for each new vtx
      // Check vtx_ids are ordered
      VertexIds sorted_vtx_ids = vtx_ids;//这个三角形对应的全局mesh 顶点序号
      std::sort(sorted_vtx_ids.begin(), sorted_vtx_ids.end());

      for (const auto& sorted_vtx_id : sorted_vtx_ids) {
        if (sorted_vtx_id >= static_cast<size_t>(adjacency_matrix_.rows)) {
          // Non-existing vertex! Add row/col. Careful! vtx_ids ordering
          // matters! First order them!
          CHECK(adjacency_matrix_.rows != 0 && adjacency_matrix_.cols != 0);
          cv::Mat row = cv::Mat::zeros(1, adjacency_matrix_.cols, CV_8UC1);
          cv::vconcat(adjacency_matrix_, row, adjacency_matrix_);

          cv::Mat col = cv::Mat::zeros(adjacency_matrix_.rows, 1, CV_8UC1);
          cv::hconcat(adjacency_matrix_, col, adjacency_matrix_);
        }
        CHECK_LT(sorted_vtx_id, adjacency_matrix_.rows);
      }
    }
    CHECK_EQ(adjacency_matrix_.rows, adjacency_matrix_.cols);

    // Update old vertices col/row
    adjacency_matrix_.at<uint8_t>(sorted_vtx_ids[0], sorted_vtx_ids[1]) = 1u;
    adjacency_matrix_.at<uint8_t>(sorted_vtx_ids[1], sorted_vtx_ids[0]) = 1u;
    adjacency_matrix_.at<uint8_t>(sorted_vtx_ids[0], sorted_vtx_ids[2]) = 1u;
    adjacency_matrix_.at<uint8_t>(sorted_vtx_ids[2], sorted_vtx_ids[0]) = 1u;
    adjacency_matrix_.at<uint8_t>(sorted_vtx_ids[1], sorted_vtx_ids[2]) = 1u;
    adjacency_matrix_.at<uint8_t>(sorted_vtx_ids[2], sorted_vtx_ids[1]) = 1u;
  } else {
    // No need to update connectivity, since the triangle is in the mesh already
    CHECK(it != face_hashes_.end());
    CHECK(face_hashes_[face_hash]);
  }
}//end function addPolygonToMesh

// Updates mesh data structures incrementally, by adding new landmark
// if there was no previous id, or updating it if it was already present.
// Provides the id of the row where the new/updated vertex is in the
// vertices_mesh data structure.
template <typename VertexPositionType>
typename Mesh<VertexPositionType>::VertexId
Mesh<VertexPositionType>::updateMeshDataStructures(
    const LandmarkId& lmk_id,
    const VertexPositionType& lmk_position,
    const VertexColorRGB& vertex_color,
    const VertexNormal& vertex_normal,
    std::map<VertexId, LandmarkId>* vertex_to_lmk_id_map,
    std::map<LandmarkId, VertexId>* lmk_id_to_vertex_id_map,
    cv::Mat* vertices_mesh,//全局mesh对应的顶点
    VertexNormals* vertices_mesh_normal,//全局mesh每个顶点对应的法向量
    cv::Mat* vertices_mesh_color) const {//全局mesh每个顶点对应的颜色


  CHECK_NOTNULL(vertex_to_lmk_id_map);
  CHECK_NOTNULL(lmk_id_to_vertex_id_map);
  CHECK_NOTNULL(vertices_mesh);
  CHECK_NOTNULL(vertices_mesh_normal);
  CHECK_NOTNULL(vertices_mesh_color);
  DCHECK(!normals_computed_) << "Normals should be invalidated before...";

  const auto& lmk_id_to_vertex_map_end = lmk_id_to_vertex_id_map->end();
  const auto& vertex_it = lmk_id_to_vertex_id_map->find(lmk_id);//判断全局地图是否有landmark这个点

  VertexId row_id_vertex;//typedef size_t VertexId;
  // Check whether this landmark is already in the set of vertices of the
  // mesh.
  if (vertex_it == lmk_id_to_vertex_map_end) {//因为这个顶点对应的landmark序号在全局的mesh中并没有，因此需要向全局mesh添加这个顶点
    // New landmark, create a new entrance in the set of vertices.
    // Store 3D points in map_points_3d.
    vertices_mesh->push_back(lmk_position);
    vertices_mesh_normal->push_back(vertex_normal);
    vertices_mesh_color->push_back(vertex_color);
    row_id_vertex = vertices_mesh->rows - 1;
    // Book-keeping.
    // Store the row in the vertices structure of this new landmark id.
    (*lmk_id_to_vertex_id_map)[lmk_id] = row_id_vertex;
    (*vertex_to_lmk_id_map)[row_id_vertex] = lmk_id;
  } else {
    // Update old landmark with new position.
    // But don't update the color information... Or should we?
    //因为这个顶点对应的landmark在全局mesh中已经存在，则需要更新这个顶点的坐标， 这个顶点的法向量，这个顶点的颜色
    row_id_vertex = vertex_it->second;
    vertices_mesh->at<VertexPositionType>(row_id_vertex) = lmk_position;
    vertices_mesh_normal->at(row_id_vertex) = vertex_normal;
    vertices_mesh_color->at<VertexColorRGB>(row_id_vertex) = vertex_color;
  }
  return row_id_vertex;
}//end function updateMeshDataStructures

/* --------------------------------------------------------------------------
 */
// Get a polygon in the mesh.
// Returns false if there is no polygon.
// TODO(Toni) this is constructing polygons on the fly, but we should instead
// store polygons, not cv::Mats
template <typename VertexPositionType>
bool Mesh<VertexPositionType>::getPolygon(const size_t& polygon_idx,
                                          Polygon* polygon) const {
  CHECK_NOTNULL(polygon);
  if (polygon_idx >= getNumberOfPolygons()) {
    VLOG(10) << "Requested polygon number: " << polygon_idx
             << ". But there are only " << getNumberOfPolygons()
             << " polygons.";
    return false;
  };

  bool has_normals =
      static_cast<size_t>(vertices_mesh_.rows) == vertices_mesh_normal_.size();
  CHECK_EQ(vertices_mesh_.rows, vertices_mesh_color_.rows);
  size_t idx_in_polygon_mesh = polygon_idx * (polygon_dimension_ + 1);
  polygon->resize(polygon_dimension_);
  for (size_t j = 0; j < polygon_dimension_; j++) {
    const int32_t& row_id_pt_j =
        polygons_mesh_.at<int32_t>(idx_in_polygon_mesh + j + 1);
    CHECK(vertex_to_lmk_id_map_.find(row_id_pt_j) !=
          vertex_to_lmk_id_map_.end());
    CHECK_LT(row_id_pt_j, vertices_mesh_.rows);
    polygon->at(j) = Vertex<VertexPositionType>(
        vertex_to_lmk_id_map_.at(row_id_pt_j),
        vertices_mesh_.at<VertexPositionType>(row_id_pt_j),
        vertices_mesh_color_.at<VertexColorRGB>(row_id_pt_j),
        has_normals ? vertices_mesh_normal_.at(row_id_pt_j) : VertexNormal());
  }
  return true;
}

/* --------------------------------------------------------------------------
 */
// Retrieve a vertex of the mesh given a LandmarkId.
// Returns true if we could find the vertex with the given landmark id
// false otherwise.
template <typename VertexPosition>
bool Mesh<VertexPosition>::getVertex(const LandmarkId& lmk_id,
                                     Vertex<VertexPosition>* vertex,
                                     VertexId* vertex_id) const {
  CHECK(vertex != nullptr || vertex_id != nullptr)
      << "No output requested, are your sure you want to use this function?";
  const auto& lmk_id_to_vertex_map_end = lmk_id_to_vertex_map_.end();
  const auto& vertex_it = lmk_id_to_vertex_map_.find(lmk_id);
  if (vertex_it == lmk_id_to_vertex_map_end) {
    // We didn't find the lmk id!
    VLOG(100) << "Lmk id: " << lmk_id << " not found in mesh.";
    return false;
  } else {
    // Construct and Return the vertex.
    const VertexId& vtx_id = vertex_it->second;
    CHECK_EQ(vertices_mesh_.rows, vertices_mesh_normal_.size());
    CHECK_EQ(vertices_mesh_.rows, vertices_mesh_color_.rows);
    CHECK_LT(vtx_id, vertices_mesh_.rows);
    if (vertex_id != nullptr) *vertex_id = vtx_id;
    if (vertex != nullptr)
      *vertex = Vertex<VertexPosition>(
          vertex_to_lmk_id_map_.at(vtx_id),
          vertices_mesh_.at<VertexPosition>(vtx_id),
          vertices_mesh_color_.at<VertexColorRGB>(vtx_id),
          vertices_mesh_normal_.at(vtx_id));
    return true;  // Meaning we found the vertex.
  }
}

/* --------------------------------------------------------------------------
 */
// Retrieve per vertex normals of the mesh.
//
template <typename VertexPositionType>
void Mesh<VertexPositionType>::computePerVertexNormals() {
  CHECK_EQ(polygon_dimension_, 3) << "Normals are only valid for dim 3 meshes.";
  LOG_IF(ERROR, normals_computed_) << "Normals have been computed already...";

  size_t n_vtx = getNumberOfUniqueVertices();//mesh所有顶点的数量！！！
  std::vector<int> counts(n_vtx, 0);

  // Set all per-vertex normals in mesh to 0, since we want to average
  // per-face
  // normals.
  clearVertexNormals();
  vertices_mesh_normal_.resize(n_vtx);

  // Walk through triangles and compute averaged vertex normals.
  Polygon polygon;
  //遍历所有的三维三角形
  for (size_t i = 0; i < getNumberOfPolygons(); i++) {
    CHECK(getPolygon(i, &polygon)) << "Could not retrieve polygon.";
    CHECK_EQ(polygon.size(), 3);
    // TODO(Toni): it would be better if we could do a polygon.getNormal();
    const VertexPositionType& p1 = polygon.at(0).getVertexPosition();
    const VertexPositionType& p2 = polygon.at(1).getVertexPosition();
    const VertexPositionType& p3 = polygon.at(2).getVertexPosition();

    // Outward-facing normal.
    VertexPositionType v21(p2 - p1);
    VertexPositionType v31(p3 - p1);
    VertexNormal normal(v31.cross(v21));

    // Normalize.
    //先使用这个三角形的三个点，计算得到这个面对应的法向量
    double norm = cv::norm(normal);
    CHECK_GT(norm, 0.0);
    normal /= norm;

    // TODO(Toni): Store normals at this point on a per-face basis.

    // Sanity check
    static constexpr double epsilon = 1e-3;  // 2.5 degrees aperture.
    v21 /= cv::norm(v21);
    v31 /= cv::norm(v31);
    LOG_IF(WARNING, std::fabs(v21.ddot(v31)) >= 1.0 - epsilon)
        << "Cross product of aligned vectors.";

    // Compute per vertex averaged normals.
    /// Indices of vertices
    //std::map<LandmarkId, VertexId> lmk_id_to_vertex_map_
    const VertexId& p1_idx = lmk_id_to_vertex_map_.at(polygon.at(0).getLmkId());
    const VertexId& p2_idx = lmk_id_to_vertex_map_.at(polygon.at(1).getLmkId());
    const VertexId& p3_idx = lmk_id_to_vertex_map_.at(polygon.at(2).getLmkId());
    /// Sum of normals per vertex
    //然后增量式更新每个顶点的法向量
    vertices_mesh_normal_.at(p1_idx) =(counts.at(p1_idx) * vertices_mesh_normal_.at(p1_idx) + normal) / (counts.at(p1_idx) + 1.0);
    vertices_mesh_normal_.at(p2_idx) = counts.at(p2_idx) * vertices_mesh_normal_.at(p2_idx) + normal / (counts.at(p2_idx) + 1.0);
    vertices_mesh_normal_.at(p3_idx) = counts.at(p3_idx) * vertices_mesh_normal_.at(p3_idx) + normal / (counts.at(p3_idx) + 1.0);
    // assumes non-zero normals...
    //进行归一化操作
    vertices_mesh_normal_.at(p1_idx) /= cv::norm(vertices_mesh_normal_.at(p1_idx));
    vertices_mesh_normal_.at(p2_idx) /= cv::norm(vertices_mesh_normal_.at(p2_idx));
    vertices_mesh_normal_.at(p3_idx) /= cv::norm(vertices_mesh_normal_.at(p3_idx));
    /// Increase counts of normals added per vertex
    counts.at(p1_idx)++;
    counts.at(p2_idx)++;
    counts.at(p3_idx)++;
  }

  CHECK_EQ(counts.size(), vertices_mesh_normal_.size());
  return;
}

/* --------------------------------------------------------------------------
 */
// Retrieve a vertex of the mesh given a LandmarkId.
// Returns true if we could find the vertex with the given landmark id
// false otherwise.
// NOT THREADSAFE.
template <typename VertexPositionType>
bool Mesh<VertexPositionType>::setVertexColor(
    const LandmarkId& lmk_id,
    const VertexColorRGB& vertex_color) {
  const auto& lmk_id_to_vertex_map_end = lmk_id_to_vertex_map_.end();
  const auto& vertex_it = lmk_id_to_vertex_map_.find(lmk_id);
  if (vertex_it == lmk_id_to_vertex_map_end) {
    // We didn't find the lmk id!
    VLOG(100) << "Lmk id: " << lmk_id << " not found in mesh.";
    return false;
  } else {
    // Color the vertex.
    vertices_mesh_color_.at<VertexColorRGB>(vertex_it->second) = vertex_color;
    return true;  // Meaning we found the vertex.
  }
}

template <typename VertexPositionType>
bool Mesh<VertexPositionType>::setVertexPosition(const LandmarkId& lmk_id,
    const VertexPositionType& vertex) {
  const auto& lmk_id_to_vertex_map_end = lmk_id_to_vertex_map_.end();
  const auto& vertex_it = lmk_id_to_vertex_map_.find(lmk_id);
  if (vertex_it == lmk_id_to_vertex_map_end) {
    // We didn't find the lmk id!
    VLOG(100) << "Lmk id: " << lmk_id << " not found in mesh.";
    return false;
  } else {
    // Change the vertex position.
    vertices_mesh_.at<VertexPositionType>(vertex_it->second) = vertex;
    return true;  // Meaning we found the vertex.
  }
}

// Get a list of all lmk ids in the mesh.
template <typename VertexPositionType>
LandmarkIds Mesh<VertexPositionType>::getLandmarkIds() const {
  LandmarkIds lmk_ids(lmk_id_to_vertex_map_.size());
  size_t i = 0;
  for (LmkIdToVertexMap::const_iterator it = lmk_id_to_vertex_map_.begin();
       it != lmk_id_to_vertex_map_.end();
       ++it) {
    lmk_ids[i] = it->first;
    i++;
  }
  CHECK_EQ(i, lmk_id_to_vertex_map_.size());
  return lmk_ids;
}

template <typename VertexPositionType>
void Mesh<VertexPositionType>::getVerticesMeshToMat(
    cv::Mat* vertices_mesh) const {
  CHECK_NOTNULL(vertices_mesh);
  *vertices_mesh = vertices_mesh_.clone();
}

template <typename VertexPositionType>
void Mesh<VertexPositionType>::getPolygonsMeshToMat(
    cv::Mat* polygons_mesh) const {
  CHECK_NOTNULL(polygons_mesh);
  *polygons_mesh = polygons_mesh_.clone();
}

template <typename VertexPositionType>
cv::Mat Mesh<VertexPositionType>::getColorsMesh(const bool& safe) const {
  return safe ? vertices_mesh_color_.clone() : vertices_mesh_color_;
}

template <typename VertexPositionType>
void Mesh<VertexPositionType>::setTopology(const cv::Mat& polygons_mesh) {
  polygons_mesh_ = polygons_mesh.clone();
  // TODO(TONI) // What about adjacency matrix!!! and face_hashes!
}

// Reset all data structures of the mesh.
template <typename VertexPositionType>
void Mesh<VertexPositionType>::clearMesh() {
  vertices_mesh_ = cv::Mat(0, 1, CV_32FC3);
  vertices_mesh_normal_ = VertexNormals();
  vertices_mesh_color_ = cv::Mat(0, 0, CV_8UC3, cv::viz::Color::blue());
  polygons_mesh_ = cv::Mat(0, 1, CV_32SC1);
  adjacency_matrix_ = cv::Mat(1, 1, CV_8UC1, cv::Scalar(0u));
  face_hashes_.clear();
  vertex_to_lmk_id_map_.clear();
  lmk_id_to_vertex_map_.clear();
}

template <typename VertexPositionType>
void Mesh<VertexPositionType>::save(const std::string& filepath) const {
  cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
  fs << "vertex_to_lmk_id_map"
     << "[";
  //保存的是特征点与mesh顶点的索引关系
  for (auto&& [vertex, lmk] : vertex_to_lmk_id_map_) {
    fs << "{";
    fs << "v" << static_cast<int>(vertex) << "l" << static_cast<int>(lmk);
    fs << "}";
  }
  fs << "]";

  fs << "lmk_id_to_vertex_map"
     << "[";
  //lmk_id_to_vertex_map_ 保存的是索引
  for (auto&& [lmk, vertex] : lmk_id_to_vertex_map_) {
    fs << "{";
    fs << "l" << static_cast<int>(lmk) << "v" << static_cast<int>(vertex);
    fs << "}";
  }
  fs << "]";

  fs << "vertices_mesh" << vertices_mesh_;//顶点矩阵
  fs << "vertices_mesh_normal" << vertices_mesh_normal_;
  fs << "normals_computed" << normals_computed_;
  fs << "vertices_mesh_color" << vertices_mesh_color_;
  fs << "polygons_mesh" << polygons_mesh_;//多边形有几个顶点，每个顶点对应的顶点序号
  // fs << "adjacency_matrix" << adjacency_matrix_;//表示哪两个顶点是否相邻
  fs << "polygon_dimension" << static_cast<int>(polygon_dimension_);//一共有几个多边形
  fs << "}";
}

template <typename VertexPositionType>
void Mesh<VertexPositionType>::load(const std::string& filepath) {
  cv::FileStorage fs(filepath, cv::FileStorage::READ);
  for (const auto& pair : fs["vertex_to_lmk_id_map"]) {
    vertex_to_lmk_id_map_[static_cast<int>(pair["v"])] = static_cast<int>(pair["l"]);
  }

  for (const auto& pair : fs["lmk_id_to_vertex_map"]) {
    lmk_id_to_vertex_map_[static_cast<int>(pair["l"])] = static_cast<int>(pair["v"]);
  }

  fs["vertices_mesh"] >> vertices_mesh_;
  fs["vertices_mesh_normal"] >> vertices_mesh_normal_;
  fs["normals_computed"] >> normals_computed_;
  fs["vertices_mesh_color"] >> vertices_mesh_color_;
  fs["polygons_mesh"] >> polygons_mesh_;
  fs["adjacency_matrix"] >> adjacency_matrix_;
  CHECK_EQ(polygon_dimension_, static_cast<int>(fs["polygon_dimension"]));
}

// explicit instantiations
template class Mesh<Vertex2D>;
template class Mesh<Vertex3D>;

}  // namespace VIO
