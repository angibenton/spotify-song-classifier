module Np = Np.Numpy
open Core
open Lwt.Syntax
open Spotify 

(* TESTER FOR THE SPOTIFY MODULE *)

type song = {name: string; sid: string; features_vector: Np.Ndarray.t;}

type playlist = {name: string; pid: string; features_matrix: Np.Ndarray.t;}

module type Spotify = sig 
  (* Generate a (PROMISE OF) new api token to be used for song_of_id and playlist_of_id. Expires in one hour. *)
  val get_new_api_token : _ -> string Lwt.t
  (* Use a song id & access token to query spotify for song data and convert the result to (PROMISE OF) a song object *)
  val song_of_id : string -> string -> playlist Lwt.t
  (* Use a playlist id & access token to query spotify for playlist data and convert the result to (PROMISE OF) a playlist object *)
  val playlist_of_id : string -> string-> playlist Lwt.t
end

let print_song (s: song): _ = 
  printf "name: %s\n" s.name;
  printf "id: %s\n" s.sid;
  printf "features: %s\n" @@ Np.Ndarray.to_string s.features_vector; 
;;

let print_playlist (p: playlist): _ = 
  printf "name: %s\n" p.name;
  printf "id: %s\n" p.pid;
  printf "features: %s\n" @@ Np.Ndarray.to_string p.features_matrix; 
;;

let () =
  let p = 
    let%lwt token = Spotify.get_new_api_token () in
    Lwt.return @@ print_endline ("Token: " ^ token);
  in
  Lwt_main.run p;
;;