open Spotify
open Numpy_helper

type confusion_matrix = {tp: int; fp: int; tn: int; fn: int}
type dataset = {pos_train: playlist; pos_valid: playlist; 
pos_test: playlist; neg_train: playlist; 
neg_valid: playlist; neg_test: playlist; 
shift: Np.Ndarray.t; scale: Np.Ndarray.t}

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
  val preprocess_features: t -> (float * float) list
end 

module type Classification = sig
  type t
  val randomize : playlist -> playlist
  val balance_classes : (playlist * playlist) -> (playlist * playlist)
  val normalize : (playlist * playlist) -> (playlist * playlist * ((float * float) list))
  val standardize : (playlist * playlist) -> (playlist * playlist * ((float * float) list))
  val split : (playlist * playlist) -> float -> float -> (float * float) list -> dataset
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

module Classification (Classifier: Model) : (Classification with type t = Classifier.t)