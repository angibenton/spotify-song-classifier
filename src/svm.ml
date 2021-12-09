open Numpy_helper
open Sklearn.Svm
open Core
open Machine_learning
open Spotify

module SVM_Model = struct
  (* the model *)
  type t = {hyperplane: Np.Ndarray.t; intercept: float; 
            class1: string; class2: string; shift: Np.Ndarray.t; scale: Np.Ndarray.t}

  type hyperparameters = {reg: float; shift: Np.Ndarray.t; scale: Np.Ndarray.t}

  (* save a model into a file with a give filename *)
  let save (svc: t) (file: string) : unit = 
    let f = Stdio.Out_channel.create file
    in match svc with |
      {hyperplane; intercept; class1; class2; shift; scale} 
      -> Stdio.Out_channel.output_string f 
           (class1 ^ "\n" ^ class2 ^ "\n" ^ (vector_to_string shift |> fun s -> if String.is_empty s then "none" else s) ^ "\n" 
            ^ (vector_to_string scale |> fun s -> if String.is_empty s then "none" else s) ^ "\n" ^ Float.to_string intercept ^ "\n" ^ 
            vector_to_string hyperplane);
      Stdio.Out_channel.flush f;
      Stdio.Out_channel.close f


  (* open up a model file with a given filename, parse a model object from it *)
  let load (file: string) : t =
    match Stdio.In_channel.read_lines file with 
    | class1 :: class2 :: scale_arr :: shift_arr :: intercept :: hyper_arr :: [] 
      -> {hyperplane = txt_to_vec hyper_arr; intercept = Float.of_string intercept; 
          class1; class2; shift = txt_to_vec shift_arr; scale = txt_to_vec scale_arr}
    | _ -> failwith "improper file formatting"

  (* train a binary classifier on two playlists represented as tensors *)
  let train (c: hyperparameters) (p1: playlist) (p2: playlist) : t =
    let x, y = match p1, p2 with
      | {features_matrix = features1; _}, {features_matrix = features2; _} 
        -> Array.init (Np.size ~axis:0 features1) ~f:(fun _ -> 1) |> fun (arr1) 
           -> (Np.append ~axis:0 ~arr:features1 () ~values:features2), 
              (Array.append arr1 @@ Array.init (Np.size ~axis:0 features2) ~f:(fun _ -> -1) 
               |> Np.Ndarray.of_int_array)
    in let clf = LinearSVC.create ~c:c.reg ~dual:false () 
    in match p1, p2 with 
    | {name = class1; _}, {name = class2; _} 
      -> LinearSVC.fit clf ~x ~y 
         |> fun svc -> {hyperplane = (LinearSVC.coef_ svc); 
                        intercept=(Array.get (Np.Ndarray.to_float_array 
                                              @@ LinearSVC.intercept_ svc) 0); 
                                              class1; class2; shift = c.shift; 
                                              scale = c.scale}

  let tune (all_c: hyperparameters list) (p1: playlist) (p2: playlist) : t list =
    List.map all_c ~f:(fun c -> train c p1 p2)

  let predict_score (hyperplane: Np.Ndarray.t) (intercept: float) (features: Np.Ndarray.t) : float =
    let normalize = Np.square hyperplane |> Np.sum |> Np.Ndarray.to_float_array
                    |> fun (arr) -> Array.get arr 0 |> Float.sqrt
    in Float.(/) 
      (Float.(+) intercept 
       @@ Array.get (Np.Ndarray.to_float_array 
                     @@ Np.dot ~b:(Np.reshape ~newshape:[(Np.Ndarray.size features);] features) 
                       (Np.reshape ~newshape:[(Np.Ndarray.size hyperplane);] hyperplane )) 0) normalize

  let predict (svc: t) (features: Np.Ndarray.t) : bool =
    match svc with 
    | {hyperplane; intercept; _} ->Float.(<=) 0. 
      @@ predict_score hyperplane intercept features

  let classes (svc: t) : string * string = 
    match svc with
    | {class1; class2; _} -> (class1, class2)

  let equal (svm1: t) (svm2: t) : bool =
    Array.equal (fun f1 f2 -> Float.(-) f1 f2 
                              |> Float.abs 
                              |>  Float.(>) 0.001) (Np.Ndarray.to_float_array svm1.hyperplane) 
    @@ Np.Ndarray.to_float_array svm2.hyperplane
    && Float.(=) svm1.intercept svm2.intercept
    && String.(=) svm1.class1 svm2.class1
    && String.(=) svm1.class2 svm2.class2

  let preprocess_features (svm: t) : (float * float) list = 
    List.zip_exn (Np.Ndarray.to_float_array svm.shift |> Array.to_list)
    (Np.Ndarray.to_float_array svm.scale |> Array.to_list)
end 

module SVM_Classification = Classification(SVM_Model)