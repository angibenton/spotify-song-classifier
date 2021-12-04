module Np = Np.Numpy
open Core
open Spotify 

(* TESTER FOR THE SPOTIFY MODULE *)

(* TODO actual tests *)

let print_song (s: Spotify.song): _ = 
  printf "name: %s\n" s.name;
  printf "id: %s\n" s.sid;
  printf "features: %s\n" @@ Np.Ndarray.to_string s.features_vector; 
;;

let print_playlist (p: Spotify.playlist): _ = 
  printf "name: %s\n" p.name;
  printf "id: %s\n" p.pid;
  printf "features: %s\n" @@ Np.Ndarray.to_string p.features_matrix; 
;;


let () =
  let p = 
    let%lwt token = Spotify_api.get_new_api_token () in
    let%lwt song = Spotify_api.song_of_id "6L89mwZXSOwYl76YXfX13s" token in 
    let%lwt playlist = Spotify_api.playlist_of_id "37i9dQZF1DX4laDTTxplAf" token in 
    print_endline ("Token: " ^ token);
    print_playlist playlist; 
    print_song song; 
    Lwt.return ()
  in
  Lwt_main.run p;
;;