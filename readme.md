<h1 align="center"> Gradient Boosting in C++ </h1>

# Data Layer

Statistical algorithms require a concrete way to store the training
data. Here the dataset is represented as a matrix
$`X \in {\mathbb{R}}^{N \times p}`$ and a target vector
$`Y \in {\mathbb{R}}^{N}`$, where $`N`$ is the number of observations
and $`p`$ is the number of features.

## Data Matrix

$`X`$ is stored as a flat `std::vector<double>` with row-major index
mapping $`f(i,j) = ic \cdot p + j`$:

<div align="center">

|  |  |  |
|:---|:---|:---|
| Matrix dimensions | $`N \times p`$ | `size_t rows, cols;` |
| Elements | $`x_{i,j} \in {\mathbb{R}}`$ | `std::vector<double> data;` |
| Index mapping | $`f(i,j) \rightarrow \text{ index}`$ | `index = i * cols + j;` |

</div>

``` cpp
#include <vector>

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
```

# Supervised Learning

Given a dataset
$`D = \left\{ \left( x_{1},y_{1} \right),\ldots,\left( x_{N},y_{N} \right) \right\}`$,
the goal is to find $`\hat{f}`$ that best approximates the true
underlying relationship $`f`$. This is done by minimizing a loss
function over the training data:

``` math
\hat{f} = \mathop{\mathrm{arg\ min}}\limits_{f}\frac{1}{N}\sum_{i = 1}^{N}\mathcal{L}(y_{i},f\left( x_{i} \right))
```

## $`L_{2}`$ Loss (Mean Squared Error)

For continuous regression, the standard choice is MSE, which penalises
predictions proportionally to their squared distance from the truth:

``` math
\mathcal{L}(y,f(x)) = {\frac{1}{2}(y - f(x))}^{2}
```

The factor of $`\frac{1}{2}`$ is a convention that cancels cleanly when
differentiating.

# Decision Trees

A regression tree partitions the feature space into $`J`$ disjoint
regions $`R_{1},\ldots,R_{J}`$ and predicts a constant $`c_{j}`$ for
every observation falling into region $`R_{j}`$:

``` math
f(x) = \sum_{j = 1}^{J}c_{j}\mathbb{1}(x \in R_{j})
```

## Minimizing SSE

The goal is to find the regions and constants that minimize the total
Sum of Squared Errors:

``` math
\text{ SSE } = \sum_{j = 1}^{J}\sum_{i \in R_{j}}\left( y_{i} - c_{j} \right)^{2}
```

For a fixed region $`R_{j}`$, the optimal constant is found by
differentiating the inner sum with respect to $`c_{j}`$ and setting it
to zero:

``` math
- 2\sum_{\left( i \in R_{j} \right)\left( y_{i} - c_{j} \right)} = 0 \Rightarrow {\hat{c}}_{j} = \frac{1}{|R_{j}|}\sum_{i \in R_{j}}y_{i}
```

So $`{\hat{c}}_{j}`$ is simply the mean of the targets in $`R_{j}`$.

## Greedy Recursive Partitioning

Finding the globally optimal partition is NP-hard, so instead a greedy
top-down algorithm is used. At each node, every possible split is
considered: a split is defined by a feature index
$`k \in \left\{ 1,\ldots,p \right\}`$ and a threshold $`s`$, producing
two child regions:

``` math
R_{1}(k,s) = \left\{ i|x_{i,k} \leq s \right\}\quad\text{ and }\quad R_{2}(k,s) = \left\{ i|x_{i,k} > s \right\}
```

The algorithm selects the $`(k,s)`$ pair that minimizes the combined SSE
of the two children:

``` math
\min\limits_{k,s}\left\lbrack \sum_{i \in R_{1}(k,s)}\left( y_{i} - {\hat{c}}_{1} \right)^{2} + \sum_{i \in R_{2}(k,s)}\left( y_{i} - {\hat{c}}_{2} \right)^{2} \right\rbrack
```

This is equivalent to maximizing the variance reduction $`\Delta`$ at
the current node $`R_{m}`$:

``` math
\Delta = \text{ Var}\left( R_{m} \right) - \left\lbrack \frac{|R_{1}|}{|R_{m}|}\text{Var}\left( R_{1} \right) + \frac{|R_{2}|}{|R_{m}|}\text{Var}\left( R_{2} \right) \right\rbrack
```

Since $`\text{Var}\left( R_{m} \right)`$ is fixed for a given node,
maximizing $`\Delta`$ is the same as minimizing the weighted sum of
child variances.

# C++ Implementation of the Regression Tree

## Node Structure

The tree is stored as a flat `std::vector<Node>`, where parent nodes
reference their children by vector index. This avoids pointer-based tree
structures and keeps memory contiguous.

|               |          |                                                   |
|:--------------|:---------|:--------------------------------------------------|
| `feature_idx` | `int`    | Splitting feature $`k`$; $`- 1`$ indicates a leaf |
| `threshold`   | `double` | Split threshold $`s`$                             |
| `prediction`  | `double` | Leaf constant $`{\hat{c}}_{j}`$                   |
| `left_child`  | `int`    | Index of child where $`x_{i,k} \leq s`$           |
| `right_child` | `int`    | Index of child where $`x_{i,k} > s`$              |

``` cpp
struct Node {
    int feature_idx = -1;
    double threshold = 0.0;
    double prediction = 0.0;
    int left_child = -1;
    int right_child = -1;

    Node(double pred) : prediction(pred) {}

    bool is_leaf() const { return feature_idx == -1; }
};
```

## $`O\left( N\log N \right)`$ Split Search

For each feature column $`k`$, the algorithm sorts observations by
$`x_{k}`$ in $`O\left( N\log N \right)`$ time. It then sweeps through
the sorted order, maintaining running sums to evaluate the SSE of each
candidate split in $`O(1)`$ per step.

The key identity is:

``` math
\sum_{i \in R}\left( y_{i} - \overline{y} \right)^{2} = \sum_{i \in R}y_{i}^{2} - \frac{1}{|R|}\left( \sum_{i \in R}y_{i} \right)^{2}
```

This lets SSE be updated incrementally as observations shift from the
right child to the left, rather than being recomputed from scratch. The
full split search over all $`p`$ features costs
$`O\left( pN\log N \right)`$ per node.

``` cpp
struct SplitResult {
    int feature_idx = -1;
    double threshold = 0.0;
    double best_sse = std::numeric_limits<double>::infinity();
};

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
```

The threshold is set to the midpoint between adjacent sorted values, so
new observations can be compared cleanly against it.

## Tree Construction and Stopping Criteria

The tree is built recursively. At each recursive call, `find_best_split`
is invoked on the current node’s index subset. If a valid split is
found, two child nodes are pushed onto the `std::vector<Node>` and the
function recurses on each.

Recursion stops under the following conditions:

- **Maximum depth reached:** the depth parameter equals `max_depth`.

- **Insufficient observations:** fewer than 2 samples remain at the
  node, making a split impossible.

- **No valid split found:** all observations share the same feature
  value (the split search returns `feature_idx == -1`).

When recursion stops, the active node retains its initial `prediction`
value (set at construction to the mean of its assigned targets), and is
treated as a leaf via `is_leaf()`.

``` cpp
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
```

## Prediction

To predict for a single observation $`x_{i}`$, the tree is traversed
from the root by following left or right branches based on threshold
comparisons, until a leaf is reached:

``` cpp
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
```

# Limitations of a Single Tree

A decision tree grown to sufficient depth can perfectly fit any training
dataset. This is actually a problem: a single observation changing can
alter the root split, cascading into a structurally different tree. The
model has high variance.

Ensemble methods address this. Gradient boosting takes a sequential
additive approach: instead of growing one deep tree, it combines many
shallow, constrained trees (weak learners), each of which corrects the
errors of its predecessors.

# Gradient Descent in Function Space

## Classical Parameter Gradient Descent

In parametric models (e.g. linear regression, neural networks), there is
a finite weight vector $`\theta \in {\mathbb{R}}^{d}`$ to optimize. The
update rule is:

``` math
\theta_{m} = \theta_{m - 1} - \eta\nabla_{\theta}\mathcal{L}(\theta_{m - 1})
```

The gradient points in the direction of steepest ascent of
$`\mathcal{L}`$; subtracting it (scaled by learning rate $`\eta`$) moves
$`\theta`$ toward a minimum.

## From Parameter Space to Function Space

Gradient boosting has no fixed weight vector. Instead, the object being
updated is the prediction function $`F`$ itself, evaluated at each
training point. The current model’s predictions form a vector in
$`{\mathbb{R}}^{N}`$:

``` math
\hat{F} = \begin{pmatrix}
F\left( x_{1} \right) \\
F\left( x_{2} \right) \\
 \vdots \\
F\left( x_{N} \right)
\end{pmatrix}
```

The gradient of the empirical loss with respect to these predictions is
computed pointwise:

``` math
g_{i} = \frac{\partial\mathcal{L}(y_{i},F\left( x_{i} \right))}{\partial F\left( x_{i} \right)}|_{F = F_{m - 1}}
```

For $`L_{2}`$ loss, this is:

``` math
\frac{\partial\mathcal{L}(y_{i},F\left( x_{i} \right))}{\partial F\left( x_{i} \right)} = - \left( y_{i} - F\left( x_{i} \right) \right)
```

So the negative gradient is just the raw residual:

``` math
- g_{i} = y_{i} - F\left( x_{i} \right)
```

A gradient descent step in function space then looks like:

``` math
F_{m\left( x_{i} \right)} = F_{(m - 1)\left( x_{i} \right)} + \eta\left( y_{i} - F_{(m - 1)\left( x_{i} \right)} \right)
```

## Generalizing with a Weak Learner

The negative gradients $`- g_{i}`$ are only defined at the $`N`$
training points. To generalize to unseen $`x`$, a shallow decision tree
$`h_{m(x)}`$ is fitted to predict $`- g_{i}`$ from $`x_{i}`$. This
extends the gradient step to the full input space:

``` math
F_{m(x)} = F_{(m - 1)(x)} + \eta h_{m(x)}
```

The ensemble built by repeating this process is gradient descent in an
infinite-dimensional function space, with trees serving as the update
direction at each step.

# The Gradient Boosting Algorithm

## Initialization

The algorithm begins with the constant prediction that minimizes the
total loss. For $`L_{2}`$, this is the global mean:

``` math
F_{0}(x) = \mathop{\mathrm{arg\ min}}\limits_{c}\sum_{i = 1}^{N}\mathcal{L}(y_{i},c) = \frac{1}{N}\sum_{i = 1}^{N}y_{i} = \overline{y}
```

## Boosting Loop (for $`m = 1`$ to $`M`$)

Each iteration executes the following steps.

**Step 1 — Compute pseudo-residuals.** For each $`i`$, evaluate the
negative gradient of the loss at the current ensemble prediction:

``` math
r_{i,m} = - \frac{\partial\mathcal{L}(y_{i},F\left( x_{i} \right))}{\partial F\left( x_{i} \right)}|_{F = F_{m - 1}} = y_{i} - F_{(m - 1)\left( x_{i} \right)}
```

**Step 2 — Fit a weak learner.** Train a regression tree $`h_{m(x)}`$ on
the dataset $`\left( X,r_{m} \right)`$, where
$`r_{m} = \left( r_{1,m},\ldots,r_{N,m} \right)`$.

**Step 3 — Compute optimal leaf values.** For each leaf region
$`R_{j,m}`$, find the constant $`\gamma_{j,m}`$ minimizing the loss:

``` math
\gamma_{j,m} = \mathop{\mathrm{arg\ min}}\limits_{\gamma}\sum_{x_{i} \in R_{j,m}}\mathcal{L}(y_{i},F_{(m - 1)\left( x_{i} \right)} + \gamma)
```

For $`L_{2}`$ loss, this is the mean of the pseudo-residuals in the leaf
— exactly what a regression tree already computes. Step 3 is therefore
absorbed automatically into tree construction.

**Step 4 — Update the ensemble.** Shrink the new tree by learning rate
$`0 < \nu \leq 1`$ and add it to the model:

``` math
F_{m(x)} = F_{(m - 1)(x)} + \nu\sum_{j = 1}^{J_{m}}\gamma_{j,m}\mathbb{1}(x \in R_{j,m})
```

Choosing $`\nu < 1`$ prevents each tree from contributing too
aggressively. Small $`\nu`$ (e.g. 0.1) acts as regularization, reducing
overfitting at the cost of requiring more trees.

# C++ Implementation

## Model Structure

The `GradientBoostingRegressor` stores the initial prediction, the
learning rate, and the full ensemble of trees:

``` cpp
struct GradientBoostingRegressor {
    int n_estimators;
    double learning_rate;
    int max_depth;
    double initial_prediction;

    std::vector<std::vector<Node>> ensemble;

    GradientBoostingRegressor(int estimators, double lr, int depth)
        : n_estimators(estimators), learning_rate(lr),
          max_depth(depth), initial_prediction(0.0) {}

    void fit(const Matrix& X, const std::vector<double>& Y);
    double predict(const std::vector<double>& x_i) const;
    double mse(const Matrix& X, const std::vector<double>& Y) const;
};
```

## Training

``` cpp
void GradientBoostingRegressor::fit(const Matrix& X, const std::vector<double>& Y) {
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
```

Because `build_tree` stores the arithmetic mean of its target values in
each leaf, and the optimal $`\gamma_{j,m}`$ for $`L_{2}`$ loss is also
that arithmetic mean, Steps 2 and 3 are merged into a single tree-fit
call.

## Inference

``` cpp
double GradientBoostingRegressor::predict(const std::vector<double>& x_i) const {
    double final_prediction = initial_prediction;
    for (size_t m = 0; m < ensemble.size(); ++m) {
        final_prediction += learning_rate * predict_single_tree(ensemble[m], x_i);
    }
    return final_prediction;
}
```

The final prediction reconstructs
$`F_{M(x)} = F_{0} + \nu\sum_{m = 1}^{M}h_{m(x)}`$ exactly.
