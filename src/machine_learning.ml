open Numpy_helper
open Core
open Spotify

type confusion_matrix = {tp: int; fp: int; tn: int; fn: int}
type dataset = {pos_train: playlist; pos_valid: playlist; pos_test: playlist; neg_train: playlist; neg_valid: playlist; neg_test: playlist}

module type Model = sig 
  (* the model *)
  type t
  type hyperparameters

  (* save a model into a file with a give filename *)
  val save : t -> string -> unit
  (* open up a model file with a given filename, parse a model object from it *)
  val load : string -> t
  (* train a binary classifier on two playlists represented as tensors *)
  val train : hyperparameters -> playlist -> playlist -> t
  (* train many classifiers with different hyperparameter combinations *)
  val tune : hyperparameters list -> playlist -> playlist -> t list
  (* predict the class a new feature vector belongs to, true being positive class *)
  val predict : t -> Np.Ndarray.t -> bool
  (* give the class names of the model *)
  val classes: t -> string * string
  val equal: t -> t -> bool
end 

module type Classification = sig
  type t
  val randomize : playlist -> playlist
  val balance_classes : (playlist * playlist) -> (playlist * playlist)
  val normalize : (playlist * playlist) -> (playlist * playlist)
  val standardize : (playlist * playlist) -> (playlist * playlist)
  val split : (playlist * playlist) -> float -> float -> dataset
  val save_dataset : dataset -> string -> unit
  val load_dataset : string -> dataset
  (* classify a song represented by a vector into one of the two playlists *)
  val classify : t -> song -> string
  (* return the confusion matrix from testing the model on a tensor of labeled songs *)
  val test : t -> playlist -> playlist -> confusion_matrix
  (* choose the best model on the validation set based on some evaulation function *)
  val tune : t list -> playlist -> playlist -> (confusion_matrix -> float) -> t
  (* format the confusion matrix to be printed *)
  val pretty_confusion : confusion_matrix -> string
  (* calculate the accuracy of a test result confusion matrix *)
  val accuracy : confusion_matrix -> float
  (* calculate the F1 Score of a test result confusion matrix *)
  val f1_score : confusion_matrix -> float

end

module Classification (Classifier: Model) : (Classification with type t = Classifier.t) = struct
  type t = Classifier.t
  let save_dataset (d: dataset) (folder: string) : unit =
    Unix.mkdir folder; save_playlist d.pos_train @@ folder ^ "/pos_train";
    save_playlist d.pos_valid @@ folder ^ "/pos_val";
    save_playlist d.pos_test @@ folder ^ "/pos_test";
    save_playlist d.neg_train @@ folder ^ "/neg_train";
    save_playlist d.neg_valid @@ folder ^ "/neg_val";
    save_playlist d.neg_test @@ folder ^ "/neg_test"
  let load_dataset (folder: string) : dataset =
    {pos_train = load_playlist @@ folder ^ "/pos_train";
     pos_valid = load_playlist @@ folder ^ "/pos_val";
     pos_test = load_playlist @@ folder ^ "/pos_test";
     neg_train = load_playlist @@ folder ^ "/neg_train";
     neg_valid = load_playlist @@ folder ^ "/neg_val";
     neg_test = load_playlist @@ folder ^ "/neg_test"; }

  (* classify a song represented by a vector into one of the two playlists 
     true = first, false = second *)
  let classify (c: t) (s: song) : string =
    match Classifier.classes c, s with 
    | (class1, class2), {features_vector; _} 
      -> if Classifier.predict c features_vector then class1 else class2

  let split ((pos, neg): (playlist * playlist)) (valid: float) (test: float) : dataset =
    let pos_valid_bound = Np.size ~axis:0 pos.features_matrix 
                          |> Float.of_int |> Float.( * ) valid 
                          |> Float.round_nearest |> Int.of_float
    in let pos_test_bound = Np.size ~axis:0 pos.features_matrix 
                            |> Float.of_int |> Float.( * ) (Float.(+) valid test) 
                            |> Float.round_nearest |> Int.of_float
    in let pos_train_bound = Np.size ~axis:0 pos.features_matrix
    in let neg_valid_bound = Np.size ~axis:0 neg.features_matrix 
                             |> Float.of_int |> Float.( * ) valid 
                             |> Float.round_nearest |> Int.of_float
    in let neg_test_bound = Np.size ~axis:0 neg.features_matrix 
                            |> Float.of_int |> Float.( * ) (Float.(+) valid test) 
                            |> Float.round_nearest |> Int.of_float
    in let neg_train_bound = Np.size ~axis:0 neg.features_matrix

    in if 0 >= pos_valid_bound || 0 >= neg_valid_bound 
    then failwith "Validation split size too small" 
    else if pos_valid_bound >= pos_test_bound || neg_valid_bound >= neg_test_bound 
    then failwith "Test split size too small" 
    else if pos_test_bound >= pos_train_bound || neg_test_bound >= neg_train_bound 
    then failwith "Train split size too small (validation + test too big)" 
    else
      {pos_valid = replace_features pos @@ rows pos.features_matrix 0 pos_valid_bound;
       pos_test = replace_features pos @@ rows pos.features_matrix pos_valid_bound pos_test_bound;
       pos_train = replace_features pos @@ rows pos.features_matrix pos_test_bound pos_train_bound;
       neg_valid = replace_features neg @@ rows neg.features_matrix 0 neg_valid_bound;
       neg_test = replace_features neg @@ rows neg.features_matrix neg_valid_bound neg_test_bound;
       neg_train = replace_features neg @@rows neg.features_matrix neg_test_bound neg_train_bound;}

  let randomize (p: playlist) : playlist =
    List.init (Np.size ~axis:0 p.features_matrix) 
      ~f: (fun index -> rows p.features_matrix index (index + 1) 
                        |> Np.Ndarray.to_float_array)
    |> fun l -> List.iter l 
      ~f:(fun row -> Array.sort row ~compare:(fun _ _ -> (Random.int 3) - 1)); 
    List.map l ~f:(fun row 
                    -> Np.Ndarray.of_float_array row 
                       |> Np.reshape ~newshape:[1; ((Np.size ~axis:1 p.features_matrix))]) 
    |> fun l -> replace_features p @@ matrix_of_vector_list l

  let balance_classes ((pos, neg): (playlist * playlist)) : (playlist * playlist) =
    let len = Int.max (Np.size ~axis:0 pos.features_matrix) 
        (Np.size ~axis:0 neg.features_matrix)
    in (replace_features pos @@ rows pos.features_matrix 0 len,
        replace_features neg @@ rows neg.features_matrix 0 len)

  let clean_features (p:playlist) (stat: (float * float) list) (f: (float * float) -> float -> float) : playlist =
    let f (col: int) (arr: Np.Ndarray.t) (feat_stats: (float * float)) : Np.Ndarray.t = Np.append ~axis:1 ~arr 
        ~values:(Printf.printf "col: %d" col; 
                 (map_vector (columns p.features_matrix col (col + 1)) 
                    (fun old -> f feat_stats old)) 
                 |> fun v -> Np.reshape ~newshape:[Np.size ~axis:0 p.features_matrix; 1] v) ()
    in replace_features p 
    @@ List.foldi stat ~init:(Np.empty [Np.size ~axis:0 p.features_matrix; 0]) ~f 

  let arr_min (arr: float array) : float =
    Array.fold arr ~init:Float.infinity 
      ~f:(fun min elem -> if Float.(<) elem min then elem else min)

  let arr_max (arr: float array) : float =
    Array.fold arr ~init:Float.neg_infinity 
      ~f:(fun max elem -> if Float.(<) max elem then elem else max)

  let arr_mean (arr: float array) : float =
    Array.fold arr ~init:(0., 0.) 
      ~f:(fun (sum, count) elem -> (Float.(+) sum elem, Float.(+) count 1.)) 
    |> fun (sum, count) -> Float.(/) sum count

  let arr_std (arr: float array) (mean: float): float =
    Array.fold arr ~init:(0., 0.) 
      ~f:(fun (sum, count) elem 
           -> (Float.(+) sum @@ Float.square @@ Float.(-) elem mean, Float.(+) count 1.)) 
    |> fun (sum, count) -> Float.sqrt @@ Float.(/) sum count


  let normalize ((pos, neg): (playlist * playlist)) : (playlist * playlist) =
    let cols = List.init (Np.size ~axis:1 pos.features_matrix) 
        ~f:(fun col -> columns (Np.append ~axis:0 ~arr:pos.features_matrix 
                                  ~values:neg.features_matrix ()) col (col + 1) 
                       |> Np.Ndarray.to_float_array)
    in let min_max = 
         List.map cols ~f:(fun arr -> (arr_min arr, arr_max arr))
    in let to_normal ((min, max): float * float) (elem: float) : float = 
         Float.(/) (Float.(-) elem min)(Float.(-) max min)
    in (clean_features pos min_max to_normal, 
        clean_features neg min_max to_normal)

  let standardize ((pos, neg): (playlist * playlist)) : (playlist * playlist) =
    let cols = List.init (Np.size ~axis:1 pos.features_matrix) 
        ~f:(fun col -> columns (Np.append ~axis:0 ~arr:pos.features_matrix 
                                  ~values:neg.features_matrix ()) col (col + 1) 
                                  |> Np.Ndarray.to_float_array)
    in let mean_std = 
         List.map cols ~f:(fun arr -> (arr_mean arr, arr_std arr @@ arr_mean arr))
    in let to_standard ((mean, std): float * float) (elem: float) : float = 
      Float.(/) (Float.(-) elem mean) std
    in (clean_features pos mean_std to_standard, 
        clean_features neg mean_std to_standard)  

  let rec test_sample_i (c: t) (samples: Np.Ndarray.t) (pos: int) (index: int) : int =
    if (index >= Np.size ~axis:0 samples) then pos 
    else (test_sample_i c samples 
            (if Classifier.predict c @@ rows samples index (index + 1) 
              then pos + 1 else pos) (index + 1))

  let test (c: t) (pos: playlist) (neg: playlist) : confusion_matrix =
    match pos, neg with 
      {features_matrix = posFeatures; _}, {features_matrix = negFeatures; _} -> 
      let tp = test_sample_i c posFeatures 0 0
      in let fp = test_sample_i c negFeatures 0 0
      in let fn = Np.size ~axis:0 posFeatures - tp
      in let tn = Np.size ~axis:0 negFeatures - fp
      in {tp; fp; fn; tn}

  let tune (models: t list) (pos: playlist) (neg: playlist) (eval: confusion_matrix -> float) : t =
    match models with 
    | [] -> failwith "No models provided"
    | first :: rest -> List.fold ~init:(first, eval @@ test first pos neg) 
                         ~f:(fun (best, score) curr 
                              -> eval @@ test curr pos neg |> fun new_score 
                                 -> if Float.(<) score new_score then (curr, new_score) 
                                 else (best, score)) rest |> fun (best, _) -> best

  let pretty_confusion (cm: confusion_matrix) : string =

    let (tp, fp, tn, fn) = match cm with 
      | {tp; fp; tn; fn} -> (tp, fp, tn, fn)

    in let padding = (fun(_) -> ' ')
    in let left_margin = String.init 10 ~f:padding

    in let margin = left_margin ^ String.init 4 ~f:padding

    in let digits (n: int) : int =
         match n with
         | 0 -> 1
         | _ -> Float.of_int n |> Float.log10 |> fun(res) -> Int.of_float res + 1

    in let max_width = List.fold_left (tp :: fp :: tn :: fn :: [])
           ~f:(fun max curr -> if curr > max then curr else max) 
           ~init:Int.min_value |> digits

    in let spaced_int (n: int) : string =
         String.init (max_width - digits n) ~f:padding ^ Int.to_string n

    in let total_width = max_width * 2 + 7 

    in let line = String.init total_width ~f:(fun (_) -> '-') ^ "\n"

    in let row_1 = 
         let spacing = String.init ((total_width - 6) / 2)  ~f:padding
         in margin ^ spacing ^ "actual " ^ spacing ^ "\n"

    in let row_2 = 
         let size = (total_width - 6) / 3
         in let base_spacing = String.init size ~f:padding
         in let extra_spacing = String.init (size + 1) ~f:padding
         in margin ^ (match (total_width - 6) % 3 with 
             | 0 -> base_spacing ^ "pos" ^ base_spacing ^ "neg" ^ base_spacing
             | 1 -> base_spacing ^ "pos" ^ extra_spacing ^ "neg" ^ base_spacing
             | 2 -> extra_spacing ^ "pos" ^ base_spacing ^ "neg" ^ extra_spacing
             | _ -> assert false) ^ "\n"

    in let row_3 = margin ^ line
    in let row_4 = left_margin ^ "pos" ^ " | " ^ (spaced_int tp) ^ " | " ^ (spaced_int fp) ^ " |\n" 
    in let row_5 = "predicted     " ^ line
    in let row_6 = left_margin ^ "neg" ^ " | " ^ (spaced_int fn) ^ " | " ^ (spaced_int tn) ^ " |\n" 
    in let row_7 = margin ^ line
    in row_1 ^ row_2 ^ row_3 ^ row_4 ^ row_5 ^ row_6 ^ row_7

  let accuracy (cm: confusion_matrix) : float =
    match cm with 
    | {tp; tn; fp; fn;} -> Float.(/) (Int.to_float @@ tp + tn) (Int.to_float @@ tp + tn + fp + fn) 

  let f1_score (cm: confusion_matrix) : float =
    match cm with 
    | {tp; fp; fn; _} -> Float.(/) (Int.to_float tp) 
                           (Float.(+) (Int.to_float tp) @@ Float.( * ) 0.5 @@ Int.to_float @@ fp + fn) 
end