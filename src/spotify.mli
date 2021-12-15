open Numpy_helper

(* Spotify library 
  - communication with the Spotify Web API  
  - extracting relevant data from Spotify's JSON representation of songs and playlists 
  - saving songs and playlists to local files 
*)

type song = {name: string; sid: string; features_vector: Np.Ndarray.t;}

type playlist = {name: string; pid: string; features_matrix: Np.Ndarray.t;}

(* Generate a (PROMISE OF) new api token to be used for song_of_id and playlist_of_id. Expires in one hour. *)
val get_new_api_token : _ -> string Lwt.t
(* Use a song id & access token to query spotify for song data and convert the result to (PROMISE OF) a song object *)
val song_of_id : string -> string -> song Lwt.t
(* Use a playlist id & access token to query spotify for playlist data and convert the result to (PROMISE OF) a playlist object *)
val playlist_of_id : string -> string -> playlist Lwt.t
(* get a new playlist that is the same as input, but with a new features vector *)
val replace_features : playlist -> Np.Ndarray.t -> playlist 
(* save a playlist record to a file *)
val save_playlist : playlist -> string -> unit 
(* load a playlist record from a file *)
val load_playlist : string -> playlist 