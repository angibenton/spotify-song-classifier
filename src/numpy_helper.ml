module Np = Np.Numpy
open Core
let matrix_of_vector_list (vecs: Np.Ndarray.t list): Np.Ndarray.t =  
  let append_vector (cur_matrix: Np.Ndarray.t) (new_vector: Np.Ndarray.t) : Np.Ndarray.t = 
    Np.append ~axis:0 ~arr:cur_matrix () ~values:new_vector
  in
  match vecs with 
  | head :: tail -> 
    List.fold_left tail ~init:head ~f:append_vector 
  | _ -> assert false
let rows (matrix: Np.Ndarray.t) (first: int) (last: int) : Np.Ndarray.t =
  Np.Ndarray.get ~key:[Np.slice ~i:first ~j:last (); 
                       Np.slice ~i:0 ~j:(Np.size ~axis:1 matrix) ()] matrix

let columns (matrix: Np.Ndarray.t) (first: int) (last: int) : Np.Ndarray.t =
  Np.Ndarray.get ~key:[Np.slice ~i:0 ~j:(Np.size ~axis:0 matrix) (); 
                       Np.slice ~i:first ~j:last ()] matrix

let map_vector (vector: Np.Ndarray.t) (f: float -> float) : Np.Ndarray.t = 
  Np.Ndarray.to_float_array vector |> Array.map ~f |> Np.Ndarray.of_float_array 