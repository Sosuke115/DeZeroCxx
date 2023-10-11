#ifndef UTILS_
#define UTILS_

#include <cassert>
#include <cstdlib>     // for std::system
#include <filesystem>  // for filesystem operations
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

#include "NumCpp.hpp"
#include "core.h"

namespace utils {
inline std::string _dot_var(const VarPtr& v, bool verbose = false) {
  std::ostringstream oss;
  std::string name = v->name;
  if (verbose && v->data != nullptr) {
    assert(!v->name.empty());

    // v->shape().str()の改行を削除
    std::string shapeStr = v->shape().str();
    shapeStr.erase(std::remove(shapeStr.begin(), shapeStr.end(), '\n'),
                   shapeStr.end());
    oss << name << ": " << shapeStr << " double";
    name = oss.str();
    oss.str("");  // clear the content
  }
  // ノードのidとしてアドレスを用いる
  // <<をオーバーライドしてるのでvそのままはだめ
  oss << std::to_string(reinterpret_cast<uintptr_t>(&(*v))) << " [label=\""
      << name << "\", color=orange, style=filled]\n";
  return oss.str();
}

inline std::string _dot_func(const FuncPtr& f) {
  std::ostringstream oss;

  // function
  oss << std::to_string(reinterpret_cast<uintptr_t>(&(*f))) << " [label=\""
      << typeid(*f).name() << "\", color=lightblue, style=filled, shape=box]\n";

  // edge
  for (const auto& x : f->inputs_) {
    oss << reinterpret_cast<uintptr_t>(&(*x)) << " -> "
        << reinterpret_cast<uintptr_t>(&(*f)) << "\n";
  }

  for (const auto& y_weak : f->outputs_) {
    if (auto y = y_weak.lock()) {  // lock the weak_ptr
      oss << reinterpret_cast<uintptr_t>(&(*f)) << " -> "
          << reinterpret_cast<uintptr_t>(&(*y)) << "\n";
    }
  }

  return oss.str();
}

inline std::string get_dot_graph(const VarPtr& output, bool verbose = true) {
  std::ostringstream oss;
  std::vector<FuncPtr> funcs = {};
  // 処理済み関数
  std::unordered_set<FuncPtr> seen_set;

  oss << "digraph g {\n";

  auto add_func = [&funcs, &seen_set](const FuncPtr& f) {
    if (!seen_set.count(f)) {
      funcs.push_back(f);
      seen_set.insert(f);
    }
  };

  add_func(output->creator_ptr);
  oss << _dot_var(output, verbose);

  while (!funcs.empty()) {
    FuncPtr f = funcs.back();
    funcs.pop_back();
    oss << _dot_func(f);
    for (int i = 0; i < f->inputs_.size(); i++) {
      oss << _dot_var(f->inputs_[i], verbose);
      if (f->inputs_[i]->creator_ptr) {
        // １つ前の関数をリストに追加
        add_func(f->inputs_[i]->creator_ptr);
      }
    }
  }
  oss << "}";
  return oss.str();
}

inline void plot_dot_graph(const VarPtr& output, bool verbose = true,
                           const std::string& to_file = "graph.png") {
  std::string dot_graph = get_dot_graph(output, verbose);
  std::filesystem::path graph_path = "tmp_graph.dot";

  // Write dot_graph to tmp_graph.dot
  {
    std::ofstream file(graph_path);
    file << dot_graph;
  }
  std::string extension =
      to_file.substr(to_file.find_last_of(".") + 1);  // Get file extension
  std::ostringstream cmd;
  cmd << "dot " << graph_path.string() << " -T " << extension << " -o "
      << to_file;

  std::cout << cmd.str().c_str() << std::endl;
  std::system(cmd.str().c_str());
}

// ndarrayは2Dなので、それを前提に実装
// 参考にしたブロードキャストルール:
// https://note.nkmk.me/python-numpy-broadcasting/
inline nc::NdArray<double> broadcast_to(const nc::NdArray<double>& in_array,
                                        const nc::Shape& shape) {
  nc::NdArray<double> out_array;

  // 目的のshapeと行数が列数が一致している時にのみbroadcastする
  // 例: [[1,2,3], [1,2,3]] (2, 3) -> [[1,2,3], [1,2,3], [1,2,3]] (3, 3)
  // 例: [[1], [2]] (2, 1) -> [[1, 1], [2, 2]] (2, 2)
  // 例: [[1]] (1, 1) -> [[1], [1]]  (2, 1)
  assert((in_array.shape().rows == 1 && in_array.shape().cols == shape.cols &&
          in_array.shape().rows <= shape.rows) ||
         (in_array.shape().cols == 1 && in_array.shape().rows == shape.rows &&
          in_array.shape().cols <= shape.cols));

  // スカラー値
  if (in_array.shape().rows == 1 && in_array.shape().cols == 1) {
    out_array = nc::NdArray<double>(shape).fill(in_array[0]);
  }
  // 列方向のbroadcast
  else if (in_array.shape().rows == 1) {
    std::vector<std::vector<double>> mat;
    std::vector<double> row_vec = in_array.toStlVector();
    for (int i = 0; i < shape.rows; i++) mat.push_back(row_vec);
    out_array = nc::NdArray(mat);
  }
  // 行方向のbroadcast
  else if (in_array.shape().cols == 1) {
    std::vector<std::vector<double>> mat;
    std::vector<double> col_vec = in_array.toStlVector();
    for (int i = 0; i < shape.cols; i++) mat.push_back(col_vec);
    out_array = nc::NdArray(mat).transpose();
  } else {
    // broadcast必要なし
    out_array = in_array;
  }

  return out_array;
}

inline nc::NdArray<double> sum_to(const nc::NdArray<double>& in_array,
                                  const nc::Shape& shape) {
  nc::NdArray<double> out_array;
  assert((shape.rows == 1 && in_array.shape().cols == shape.cols) ||
         (shape.cols == 1 && in_array.shape().rows == shape.rows));

  // スカラー値
  if (shape.rows == 1 && shape.cols == 1) {
    out_array = in_array.sum();
  } else if (shape.rows == 1) {
    out_array = in_array.sum(nc::Axis::COL);
  } else if (shape.cols == 1) {
    out_array = in_array.sum(nc::Axis::ROW).transpose();
  } else {
    out_array = in_array;
  }
  return out_array;
}
}  // namespace utils

#endif