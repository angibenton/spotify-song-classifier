module Np = Np.Numpy
let () = 
  let n_samples, n_features = 10, 5 in
  Np.Random.seed 0;
  let y = Np.Random.uniform ~size:[n_samples] () in
  let x = Np.Random.uniform ~size:[n_samples; n_features] () in
  let open Sklearn.Svm in
  let clf = SVR.create ~c:1.0 ~epsilon:0.2 () in
  Format.printf "%a\n" SVR.pp @@ SVR.fit clf ~x ~y;
  Format.printf "%a\n" Np.pp @@ SVR.support_vectors_ clf;