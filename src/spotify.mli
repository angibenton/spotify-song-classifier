open Numpy_helper
type song = {name: string; sid: string; features_vector: Np.Ndarray.t;}

type playlist = {name: string; pid: string; features_matrix: Np.Ndarray.t;}

(* Generate a (PROMISE OF) new api token to be used for song_of_id and playlist_of_id. Expires in one hour. *)
val get_new_api_token : _ -> string Lwt.t
(* Use a song id & access token to query spotify for song data and convert the result to (PROMISE OF) a song object *)
val song_of_id : string -> string -> song Lwt.t
(* Use a playlist id & access token to query spotify for playlist data and convert the result to (PROMISE OF) a playlist object *)
val playlist_of_id : string -> string -> playlist Lwt.t
val replace_features : playlist -> Np.Ndarray.t -> playlist 
val save_playlist : playlist -> string -> unit 
val load_playlist : string -> playlist 