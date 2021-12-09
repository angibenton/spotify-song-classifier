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

let vector_to_string (vector: Np.Ndarray.t) : string =
  Np.Ndarray.to_float_array vector |> Array.map ~f:Float.to_string |> 
  Array.fold ~init:"" ~f:(fun s elem -> s ^ " " ^ elem) 

let matrix_to_string (matrix: Np.Ndarray.t) : string =
  List.fold ~init:"" ~f:(fun s row -> s ^ "\n" ^ vector_to_string row) 
    (List.init (Np.size ~axis:0 matrix) 
       ~f:(fun index -> rows matrix index (index + 1)))

let txt_to_vec (line: string) : Np.Ndarray.t =
  if String.(=) "none" line then Np.empty [0] else
  String.split ~on:' ' line 
  |> List.filter ~f:(fun (s) -> not @@ String.is_empty s)
  |> List.map ~f:Float.of_string |> Np.Ndarray.of_float_list 
  |> fun v -> Np.reshape v ~newshape:[1; Np.size v]

let txt_to_matrix (lines: string list) : Np.Ndarray.t =
  List.filter ~f:(fun s -> not @@ String.is_empty s) lines 
  |> fun rows -> List.map rows ~f:txt_to_vec
                 |> fun vecs -> matrix_of_vector_list vecs