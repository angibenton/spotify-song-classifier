module Np = Np.Numpy

open Spotify

type confusion_matrix = {tp: int; fp: int; tn: int; fn: int}

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
  (* predict the class a new feature vector belongs to, true being positive class *)
  val predict : t -> Np.Ndarray.t -> bool
  (* give the class names of the model *)
  val classes: t -> string * string
  val equal: t -> t -> bool
end 

module type Classification = sig
  type t
  (* classify a song represented by a vector into one of the two playlists *)
  val classify : t -> song -> string
  (* return the confusion matrix from testing the model on a tensor of labeled songs *)
  val test : t -> playlist -> playlist -> confusion_matrix
  (* format the confusion matrix to be printed *)
  val pretty_confusion : confusion_matrix -> string
  (* calculate the accuracy of a test result confusion matrix *)
  val accuracy : confusion_matrix -> float
  (* calculate the F1 Score of a test result confusion matrix *)
  val f1_score : confusion_matrix -> float
end

module Classification (Classifier: Model) : (Classification with type t = Classifier.t)