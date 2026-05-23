#set page(paper: "presentation-16-9", margin: (x: 2.4cm, y: 1.6cm))
#set text(font: "TeX Gyre Pagella", size: 19pt)
#set par(justify: false, leading: 0.65em)
#show raw: set text(font: "DejaVu Sans Mono", size: 14pt)
#show math.equation: set text(size: 18pt)

#let dim = rgb("#777777")

#let slide(body) = { pagebreak(weak: true); body }


#let eq(body) = { v(0.2em); align(center, body); v(0.2em) }

#let code(body) = block(
  fill: rgb("#f4f4f4"),
  radius: 2pt,
  inset: (x: 0.9em, y: 0.65em),
  width: 100%,
  body,
)

// ── 1. title ──────────────────────────────────────────────────────────────────
#slide({
  v(1fr)
  align(center)[
    #text(size: 30pt, weight: "bold")[Gradient Boosting C++-ban]
  ]
  v(1fr)
})

// ── 2. problem setting ────────────────────────────────────────────────────────
#slide({
  [Megfigyelt párok $(x, y)$, ismeretlen $f : cal(X) -> cal(Y)$, ahol $cal(X) subset.eq RR^p$, $cal(Y) subset.eq RR$.]
  v(0.4em)
  [Keressünk $hat(f) : cal(X) -> cal(Y)$-t véges mintából:]
  eq[$D = {(x_1, y_1), dots, (x_N, y_N)} subset cal(X) times cal(Y)$]
  v(0.5em)
  text(fill: dim)[A gradient boosting $hat(f)$-et sok kis _gyenge tanuló_ (sekély fa) összegeként építi fel, mindegyik az előzők hibáját korrigálja.]
})

// ── 3. data layer ─────────────────────────────────────────────────────────────
#slide({
  [$X in RR^(N times p)$ mátrix (sor = megfigyelés, oszlop = feature), $Y in RR^N$ célvektor.]
  v(0.4em)
  [Sor-főrendű tárolás: $"idx"(i,j) = i dot "cols" + j$]
  v(0.4em)
  code(raw(lang: "cpp",
    "struct Matrix {\n" +
    "    size_t rows, cols;\n" +
    "    std::vector<double> data;\n" +
    "    double& at(size_t i, size_t j) { return data[i*cols + j]; }\n" +
    "};"))
})

// ── 4. supervised learning ────────────────────────────────────────────────────
#slide({
  [_Veszteségfüggvény_ $cal(L) : cal(Y) times cal(Y) -> RR_(>=0)$. Választjuk $hat(f)$-et az átlagos veszteség minimalizálásával:]
  eq[$hat(f) = limits(op("arg min"))_f frac(1, N) sum_(i=1)^N cal(L)(y_i, f(x_i))$]
  v(0.5em)
  [$L_2$ veszteség (MSE) folytonos regresszióhoz:]
  eq[$cal(L)(y, hat(y)) = frac(1,2)(y - hat(y))^2$]
})

// ── 5. decision tree ──────────────────────────────────────────────────────────
#slide({
  [_Regressziós fa:_ $cal(X)$-et $J$ diszjunkt régióra particionálja, régiónként konstanst jósol.]
  eq[$hat(f)(x) = sum_(j=1)^J c_j bb(1)(x in R_j)$]
  v(0.5em)
  [Indexhalmaz régiónként: $I_j = {i : x_i in R_j}$]
})

// ── 6. optimal leaf constant ──────────────────────────────────────────────────
#slide({
  [Minimalizálandó össz-SSE:]
  eq[$"SSE" = sum_(j=1)^J sum_(i in R_j)(y_i - c_j)^2$]
  v(0.4em)
  [Fixált $R_j$-re $c_j$ szerint deriválva:]
  eq[$-2 sum_(i in R_j)(y_i - c_j) = 0 quad arrow.r.double quad hat(c)_j = frac(1, |I_j|) sum_(i in R_j) y_i$]
  v(0.4em)
  text(fill: dim)[A levél optimuma az ott lévő célértékek számtani közepe.]
})

// ── 7. greedy split ───────────────────────────────────────────────────────────
#slide({
  [Globális optimum NP-nehéz. Csúcsonként a legjobb $(k, s)$ párt keressük:]
  v(0.3em)
  eq[$R_1(k,s) = {x in R : x_k <= s}, quad R_2(k,s) = {x in R : x_k > s}$]
  v(0.4em)
  eq[$min_(k,s) [ sum_(i in I_1(k,s))(y_i - hat(c)_1)^2 + sum_(i in I_2(k,s))(y_i - hat(c)_2)^2 ]$]
  v(0.3em)
  text(fill: dim)[Ekvivalens a variancia-csökkenés $Delta(k,s)$ maximalizálásával.]
})

// ── 8. O(N log N) split search ────────────────────────────────────────────────
#slide({
  [Kulcs azonosság --- inkrementális momentum frissítés:]
  eq[$sum_(i in R)(y_i - macron(y))^2 = sum_(i in R) y_i^2 - frac(1,|R|)( sum_(i in R) y_i )^2$]
  v(0.4em)
  code(raw(lang: "cpp",
    "sum_left  += t;  sum_sq_left  += t*t;  n_left++;\n" +
    "sum_right -= t;  sum_sq_right -= t*t;  n_right--;\n" +
    "\n" +
    "sse = (sum_sq_left  - sum_left *sum_left /n_left)\n" +
    "    + (sum_sq_right - sum_right*sum_right/n_right);"))
  v(0.3em)
  text(fill: dim)[Teljes keresés $p$ feature felett: $O(p N log N)$ csúcsonként.]
})

// ── 9. node structure ─────────────────────────────────────────────────────────
#slide({
  code(raw(lang: "cpp",
    "struct Node {\n" +
    "    int    feature_idx = -1;  // k  (-1 = levél)\n" +
    "    double threshold   = 0.0; // s\n" +
    "    double prediction  = 0.0; // ĉⱼ\n" +
    "    int    left_child  = -1;  // vektor index\n" +
    "    int    right_child = -1;\n" +
    "};"))
  v(0.7em)
  text(fill: dim)[Pointer-mentes `std::vector<Node>`, gyerekek vektor-indexszel. Megállási feltételek: max. mélység, kevesebb mint 2 minta, vagy nincs érvényes osztás.]
})

// ── 10. limitations of a single tree ──────────────────────────────────────────
#slide({
  [Nagyon mély fa tökéletesen illeszkedik tanító adathalmazra --- magas variancia. Egyetlen adatpont megváltozása megváltoztathatja a gyökér kettéosztási pontját.]
  v(1em)
  align(center)[
    #text(fill: dim)[Megoldás:] sok sekély fa szekvenciálisan, összeadogatva
  ]
})

// ── 11. parameter vs function space ───────────────────────────────────────────
#slide({
  [_Paramétertérben_ (lineáris regresszió, neurális hhálók): véges $theta in RR^d$.]
  eq[$theta_m = theta_(m-1) - eta nabla_theta cal(L)(theta_(m-1))$]
  v(0.6em)
  [_Függvénytérben_ (boosting): nincs rögzített $theta$, maga a prediktáló $F$ a változó. Gradiens pontonként:]
  eq[$g_i = frac(partial cal(L)(y_i, F(x_i)), partial F(x_i)) |_(F=F_(m-1))$]
})

// ── 12. negative gradient = residual ──────────────────────────────────────────
#slide({
  eq[$frac(partial cal(L)(y_i, F(x_i)), partial F(x_i)) = -(y_i - F(x_i))$]
  v(0.5em)
  eq[$-g_i = y_i - F(x_i)$]
  v(0.6em)
  text(fill: dim)[$L_2$-nél a negatív gradiens a reziduum.]
  v(0.4em)
  eq[$F_m(x_i) = F_(m-1)(x_i) + eta (y_i - F_(m-1)(x_i))$]
})

// ── 13. generalizing with weak learner ────────────────────────────────────────
#slide({
  [$-g_i$ csak a megfigyelt $N$ ponton van értelmezve. Sekély fa $h_m(x)$ illeszthető $-g$-re, ez nem látott adatpontokra is kiértékelhetővé teszi:]
  eq[$F_m(x) = F_(m-1)(x) + eta h_m(x)$]
  v(0.6em)
  text(fill: dim)[Az együttes (ensemble): _gradient descent_ egy végtelen dimenziós függvénytérben, fákkal mint frissítési irányokkal.]
})

// ── 14. the algorithm ─────────────────────────────────────────────────────────
#slide({
  [Init: $F_0(x) = macron(y)$. Aztán $m = 1, dots, M$:]
  v(0.3em)
  grid(columns: (1.6em, 1fr), gutter: 0.4em,
    [1.], [pszeudo-rezidum $r_(i,m) = y_i - F_(m-1)(x_i)$ kiszámolása],
    [2.], [$h_m(x)$ sekély fa illesztése az $(X, r_m)$ adatra],
    [3.], [optimális levél-konstans $gamma_(j,m)$ kiszámolása. ez $L_2$-re automatikusan megtörténik a fa felépítése során.],
    [4.], [frissítés: $F_m(x) = F_(m-1)(x) + nu h_m(x)$],
  )
  v(0.5em)
  text(fill: dim)[$nu < 1$: tanulási ráta, regularizáció. Tipikus pl.: $nu = 0.1$.]
})

// ── 15. C++ structure ─────────────────────────────────────────────────────────
#slide({
  code(raw(lang: "cpp",
    "struct GradientBoostingRegressor {\n" +
    "    int    n_estimators;\n" +
    "    double learning_rate;\n" +
    "    int    max_depth;\n" +
    "    double initial_prediction;\n" +
    "    std::vector<std::vector<Node>> ensemble;\n" +
    "\n" +
    "    void   fit(const Matrix& X, const std::vector<double>& Y);\n" +
    "    double predict(const std::vector<double>& x_i) const;\n" +
    "};"))
})

// ── 16. training loop ─────────────────────────────────────────────────────────
#slide({
  code(raw(lang: "cpp",
    "initial_prediction = mean(Y);              // F_0\n" +
    "std::vector<double> F_m(N, initial_prediction);\n" +
    "\n" +
    "for (int m = 0; m < n_estimators; ++m) {\n" +
    "    for (i = 0; i < N; ++i)\n" +
    "        pseudo_residuals[i] = Y[i] - F_m[i];        // 1.\n" +
    "    auto tree = build_tree(X, pseudo_residuals, ...);// 2+3.\n" +
    "    ensemble.push_back(tree);\n" +
    "    for (i = 0; i < N; ++i)\n" +
    "        F_m[i] += learning_rate * predict_single_tree(tree, x_i);// 4.\n" +
    "}"))
})

// ── 17. inference ─────────────────────────────────────────────────────────────
#slide({
  eq[$F_M(x) = F_0 + nu sum_(m=1)^M h_m(x)$]
  v(0.5em)
  code(raw(lang: "cpp",
    "double predict(const std::vector<double>& x_i) const {\n" +
    "    double pred = initial_prediction;\n" +
    "    for (const auto& tree : ensemble)\n" +
    "        pred += learning_rate * predict_single_tree(tree, x_i);\n" +
    "    return pred;\n" +
    "}"))
})

// ── 18. closing ───────────────────────────────────────────────────────────────
#slide({
  v(1fr)
  align(center, text(size: 22pt, fill: dim)[Köszi a figyelmet + kérdések])
  v(1fr)
})
