#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
#include <array>

struct Matrix {
    size_t rows;
    size_t cols;
    std::vector<double> data;

    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0) {}

    double& at(size_t row, size_t col) {
        return data[row * cols + col];
    }

    const double& at(size_t row, size_t col) const {
        return data[row * cols + col];
    }
};

struct Node {
    int feature_idx = -1;
    double threshold = 0.0;
    double prediction = 0.0;
    int left_child = -1;
    int right_child = -1;

    Node(double pred) : prediction(pred) {}

    bool is_leaf() const {
        return feature_idx == -1;
    }
};

struct SplitResult {
    int feature_idx = -1;
    double threshold = 0.0;
    double best_sse = std::numeric_limits<double>::infinity();
};

// O(N log N)
SplitResult find_best_split(const Matrix& X, const std::vector<double>& y, const std::vector<int>& indices) {
    SplitResult best_split;
    size_t n = indices.size();
    if (n < 2) return best_split;

    for (size_t k = 0; k < X.cols; ++k) {
        std::vector<std::pair<double, int>> feature_vals(n);
        for (size_t i = 0; i < n; ++i) {
            feature_vals[i] = {X.at(indices[i], k), indices[i]};
        }

        std::sort(feature_vals.begin(), feature_vals.end());

        double sum_left = 0.0, sum_right = 0.0;
        double sum_sq_left = 0.0, sum_sq_right = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double target = y[feature_vals[i].second];
            sum_right += target;
            sum_sq_right += target * target;
        }

        size_t n_left = 0, n_right = n;

        for (size_t i = 0; i < n - 1; ++i) {
            double target = y[feature_vals[i].second];

            sum_left += target;
            sum_sq_left += target * target;
            n_left++;

            sum_right -= target;
            sum_sq_right -= target * target;
            n_right--;

            if (feature_vals[i].first == feature_vals[i+1].first) continue;

            double sse_left = sum_sq_left - (sum_left * sum_left) / n_left;
            double sse_right = sum_sq_right - (sum_right * sum_right) / n_right;
            double total_sse = sse_left + sse_right;

            if (total_sse < best_split.best_sse) {
                best_split.best_sse = total_sse;
                best_split.feature_idx = k;
                best_split.threshold = (feature_vals[i].first + feature_vals[i+1].first) / 2.0; 
            }
        }
    }
    return best_split;
}

void build_tree_recursive(const Matrix& X, const std::vector<double>& y, const std::vector<int>& indices, 
        int depth, int max_depth, std::vector<Node>& tree, int node_idx) {
    if (depth >= max_depth || indices.size() < 2) return;

    SplitResult best = find_best_split(X, y, indices);
    if (best.feature_idx == -1) return; 

    tree[node_idx].feature_idx = best.feature_idx;
    tree[node_idx].threshold = best.threshold;

    std::vector<int> left_idx, right_idx;
    double left_sum = 0.0, right_sum = 0.0;

    for (int idx : indices) {
        if (X.at(idx, best.feature_idx) <= best.threshold) {
            left_idx.push_back(idx);
            left_sum += y[idx];
        } else {
            right_idx.push_back(idx);
            right_sum += y[idx];
        }
    }

    if (!left_idx.empty()) {
        tree.push_back(Node(left_sum / left_idx.size()));
        tree[node_idx].left_child = tree.size() - 1;
        build_tree_recursive(X, y, left_idx, depth + 1, max_depth, tree, tree[node_idx].left_child);
    }

    if (!right_idx.empty()) {
        tree.push_back(Node(right_sum / right_idx.size()));
        tree[node_idx].right_child = tree.size() - 1;
        build_tree_recursive(X, y, right_idx, depth + 1, max_depth, tree, tree[node_idx].right_child);
    }
}

std::vector<Node> build_tree(const Matrix& X, const std::vector<double>& y, const std::vector<int>& indices, int max_depth) {
    double sum = 0.0;
    for (int idx : indices) sum += y[idx];

    std::vector<Node> tree;
    tree.push_back(Node(sum / indices.size())); // Root node
    build_tree_recursive(X, y, indices, 0, max_depth, tree, 0);
    return tree;
}

double predict_single_tree(const std::vector<Node>& tree, const std::vector<double>& x_i) {
    int curr = 0;
    while (!tree[curr].is_leaf()) {
        if (x_i[tree[curr].feature_idx] <= tree[curr].threshold) {
            curr = tree[curr].left_child;
        } else {
            curr = tree[curr].right_child;
        }
    }
    return tree[curr].prediction;
}

struct GradientBoostingRegressor {
    int n_estimators;          
    double learning_rate;      
    int max_depth;             
    double initial_prediction; 

    std::vector<std::vector<Node>> ensemble;

    GradientBoostingRegressor(int estimators, double lr, int depth) 
        : n_estimators(estimators), learning_rate(lr), max_depth(depth), initial_prediction(0.0) {}

    void fit(const Matrix& X, const std::vector<double>& Y) {
        size_t N = Y.size();

        double sum = 0.0;
        for (double y : Y) sum += y;
        initial_prediction = sum / N;

        std::vector<double> F_m(N, initial_prediction);
        std::vector<double> pseudo_residuals(N, 0.0);

        std::vector<int> all_indices(N);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        for (int m = 0; m < n_estimators; ++m) {
            for (size_t i = 0; i < N; ++i) {
                pseudo_residuals[i] = Y[i] - F_m[i];
            }

            std::vector<Node> tree = build_tree(X, pseudo_residuals, all_indices, max_depth);
            ensemble.push_back(tree);

            for (size_t i = 0; i < N; ++i) {
                std::vector<double> x_i(X.cols);
                for(size_t j = 0; j < X.cols; ++j) {
                    x_i[j] = X.at(i, j);
                }

                double tree_pred = predict_single_tree(tree, x_i);
                F_m[i] += learning_rate * tree_pred; 
            }
        }
    }

    double predict(const std::vector<double>& x_i) const {
        double final_prediction = initial_prediction;
        for (size_t m = 0; m < ensemble.size(); ++m) {
            final_prediction += learning_rate * predict_single_tree(ensemble[m], x_i);
        }
        return final_prediction;
    }
    double mse(const Matrix& X, const std::vector<double>& Y) const {
        double err = 0.0;
        for (size_t i = 0; i < X.rows; ++i) {
            std::vector<double> x_i(X.cols);
            for (size_t j = 0; j < X.cols; ++j)
                x_i[j] = X.at(i, j);
            double r = Y[i] - predict(x_i);
            err += r * r;
        }
        return err / X.rows;
    }
};

void print_row(double pred, double truth) {
    std::cout << std::fixed << std::setprecision(3)
              << "  predicted: " << std::setw(8) << pred
              << "   true: "     << std::setw(8) << truth
              << "   error: "    << std::setw(8) << std::abs(pred - truth) << "\n";
}

// 1. y = x² (1 feature, 5 points)
void test_parabola() {
    std::cout << "=== y = x^2,  x in {0..4} ===\n";
    Matrix X(5, 1);
    std::vector<double> y = {0, 1, 4, 9, 16};
    for (size_t i = 0; i < 5; ++i) X.at(i,0) = i;
    GradientBoostingRegressor gbm(50, 0.1, 3);
    gbm.fit(X, y);
    for (int x = 0; x <= 4; ++x)
        print_row(gbm.predict({(double)x}), x*x);
}

// 2. y = x0 + x1 (2 features, additive)
// Points: (0,0)->0, (1,0)->1, (0,1)->1, (1,1)->2, (2,1)->3, (2,2)->4
void test_additive() {
    std::cout << "\n=== y = x0 + x1 ===\n";
    std::vector<std::pair<double,double>> pts = {{0,0},{1,0},{0,1},{1,1},{2,1},{2,2}};
    std::vector<double> y;
    Matrix X(pts.size(), 2);
    for (size_t i = 0; i < pts.size(); ++i) {
        X.at(i,0) = pts[i].first;
        X.at(i,1) = pts[i].second;
        y.push_back(pts[i].first + pts[i].second);
    }
    GradientBoostingRegressor gbm(50, 0.1, 3);
    gbm.fit(X, y);
    for (size_t i = 0; i < pts.size(); ++i)
        print_row(gbm.predict({pts[i].first, pts[i].second}), y[i]);
}

// 3. y = x0 * x1 (2 features, interaction)
// Points: (1,1)->1, (1,2)->2, (2,2)->4, (2,3)->6, (3,3)->9
void test_product() {
    std::cout << "\n=== y = x0 * x1 ===\n";
    std::vector<std::pair<double,double>> pts = {{1,1},{1,2},{2,2},{2,3},{3,3}};
    std::vector<double> y;
    Matrix X(pts.size(), 2);
    for (size_t i = 0; i < pts.size(); ++i) {
        X.at(i,0) = pts[i].first;
        X.at(i,1) = pts[i].second;
        y.push_back(pts[i].first * pts[i].second);
    }
    GradientBoostingRegressor gbm(50, 0.1, 3);
    gbm.fit(X, y);
    for (size_t i = 0; i < pts.size(); ++i)
        print_row(gbm.predict({pts[i].first, pts[i].second}), y[i]);
}

// 4. y = x0  (3 features, only x0 matters)
// Checks that irrelevant features x1,x2 don't corrupt the fit.
// Points: x0 in {1..5}, x1 and x2 are noise 
void test_irrelevant_features() {
    std::cout << "\n=== y = x0  (x1, x2 irrelevant) ===\n";
    // x1, x2 vary but y depends only on x0
    std::vector<std::array<double,3>> pts = {
        {1, 99, 42}, {2, 7, 13}, {3, 55, 0}, {4, 23, 77}, {5, 1, 100}
    };
    std::vector<double> y;
    Matrix X(pts.size(), 3);
    for (size_t i = 0; i < pts.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) X.at(i,j) = pts[i][j];
        y.push_back(pts[i][0]);
    }
    GradientBoostingRegressor gbm(50, 0.1, 3);
    gbm.fit(X, y);
    for (size_t i = 0; i < pts.size(); ++i)
        print_row(gbm.predict({pts[i][0], pts[i][1], pts[i][2]}), y[i]);
}

// 5. y = x0 + 2*x1 - x2   (3 features, linear combination)
void test_linear_combo() {
    std::cout << "\n=== y = x0 + 2*x1 - x2 ===\n";
    // (x0, x1, x2) -> y
    // (1,1,1)->2, (2,1,0)->4, (0,2,1)->3, (3,0,3)->0, (1,3,2)->5
    std::vector<std::array<double,3>> pts = {
        {1,1,1}, {2,1,0}, {0,2,1}, {3,0,3}, {1,3,2}
    };
    std::vector<double> y;
    Matrix X(pts.size(), 3);
    for (size_t i = 0; i < pts.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) X.at(i,j) = pts[i][j];
        y.push_back(pts[i][0] + 2*pts[i][1] - pts[i][2]);
    }
    GradientBoostingRegressor gbm(50, 0.1, 3);
    gbm.fit(X, y);
    for (size_t i = 0; i < pts.size(); ++i)
        print_row(gbm.predict({pts[i][0], pts[i][1], pts[i][2]}), y[i]);
}

int main() {
    test_parabola();
    test_additive();
    test_product();
    test_irrelevant_features();
    test_linear_combo();
}
