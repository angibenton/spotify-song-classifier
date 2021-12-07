module Np = Np.Numpy
open Core
open Spotify
open OUnit2

(* TESTER FOR THE SPOTIFY MODULE *)
(* makeshift (sequential!) test suite because the dune test setup seems to cause issues with asynch *)

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

let number_of_features = 13;;

let test_playlist_id = "01keRwFGHF7Rw1wnPwbyB1";;
let test_song_id_vienna = "4U45aEWtQhrm8A5mxPaFZ7";;
let test_song_id_chain = "5e9TFTbltYBg2xThimr0rU";;
let test_song_id_wish = "1HzDhHApjdjXPLHF6GGYhu";;
let test_song_id_skylines = "3VqJUdav5hRAEAvppYIVnT";;
let expired_token = "BQCeq-iZTX3AWwOhNVt9fM3vZBicMy8pVSINCAjgJ-Yge4PnF-OPae-SMVLJdzk-YiM10yFxtnZrntcxo2w";;
let ill_formed_token = "MyToken123";;
let ill_formed_song_id = "MySongId123";;
let ill_formed_playlist_id = "MySongId123";;


(* -------- TESTS: GOOD BEHAVIOR --------- *)

let test_get_token _ = 
  let test_monadic = 
    let%lwt token = get_new_api_token () in
    (* check if token is well-formed *)
    assert_equal 83 @@ String.length token;
    assert_equal token @@ String.filter ~f:(fun c -> Char.is_alphanum c || Char.(=) c '-' || Char.(=) c '_') token;
    Lwt.return (); 
  in
  Lwt_main.run test_monadic;  
;;

let test_get_song _ = 
  let test_monadic = 
    let%lwt token = get_new_api_token () in
    let%lwt chain = song_of_id test_song_id_chain token in
    let%lwt skylines = song_of_id test_song_id_skylines token in
    assert_equal "The Chain - 2004 Remaster" @@ chain.name;
    assert_equal "Skylines and Turnstiles" @@ skylines.name;
    assert_equal test_song_id_chain @@ chain.sid;
    assert_equal test_song_id_skylines @@ skylines.sid;
    assert_equal [|1; number_of_features;|] @@ Np.shape chain.features_vector;
    assert_equal [|1; number_of_features;|] @@ Np.shape skylines.features_vector;
    Lwt.return (); 
  in
  Lwt_main.run test_monadic;  
;;


let test_get_playlist _ = 
  let test_monadic = 
    let%lwt token = get_new_api_token () in
    let%lwt test_playlist = playlist_of_id test_playlist_id token in
    assert_equal "TestPlaylist" @@ test_playlist.name;
    assert_equal test_playlist_id @@ test_playlist.pid;
    (* this is my own playlist, definitely has 3 songs *)
    assert_equal [|3; number_of_features;|] @@ Np.shape test_playlist.features_matrix;
    Lwt.return (); 
  in
  Lwt_main.run test_monadic;  
;;



(* -------- TESTS: EXCEPTIONS --------- *)

(* A lot of the exception testing is just indirectly confirming that there is no convention at all for spotify API error messages *)

let test_bad_token_exceptions _ = 
  let req_song_bad_token _ = Lwt_main.run @@ song_of_id test_song_id_vienna ill_formed_token in
  assert_raises (Failure ("request features for song (id = " ^ test_song_id_vienna ^ ") failed: 401 Invalid access token")) req_song_bad_token;
  let req_song_expired_token _ = Lwt_main.run @@ song_of_id test_song_id_vienna expired_token in
  assert_raises (Failure ("request features for song (id = " ^ test_song_id_vienna ^ ") failed: 401 The access token expired")) req_song_expired_token;
  let req_play_bad_token _ = Lwt_main.run @@ playlist_of_id test_playlist_id ill_formed_token in
  assert_raises (Failure ("request metadata for playlist (id = " ^ test_playlist_id ^ ") failed: 401 Invalid access token")) req_play_bad_token;
  let req_play_expired_token _ = Lwt_main.run @@ playlist_of_id test_playlist_id expired_token in
  assert_raises (Failure ("request metadata for playlist (id = " ^ test_playlist_id ^ ") failed: 401 The access token expired")) req_play_expired_token;
;;


let test_bad_id_exceptions _ = 

  (* ill-formed ids *)

  let req_song_bad_id _ = 
    Lwt_main.run begin
      let%lwt token = get_new_api_token () in
      song_of_id ill_formed_song_id token
    end
  in
  assert_raises (Failure ("request features for song (id = " ^ ill_formed_song_id ^ ") failed: 400 invalid request")) req_song_bad_id; 

  let req_playlist_bad_id _ = 
    Lwt_main.run begin
      let%lwt token = get_new_api_token () in
      playlist_of_id ill_formed_playlist_id token
    end
  in
  assert_raises (Failure ("request metadata for playlist (id = " ^ ill_formed_playlist_id ^ ") failed: 404 Invalid playlist Id")) req_playlist_bad_id; 

  (* requesting a song with the id of a playlist and vice versa *)

  let req_song_playlist_id _ = 
    Lwt_main.run begin
      let%lwt token = get_new_api_token () in
      song_of_id test_playlist_id token
    end
  in
  assert_raises (Failure ("request features for song (id = " ^ test_playlist_id ^ ") failed: 404 analysis not found")) req_song_playlist_id; 

  let req_playlist_song_id _ = 
    Lwt_main.run begin
      let%lwt token = get_new_api_token () in
      playlist_of_id test_song_id_skylines token
    end
  in
  assert_raises (Failure ("request metadata for playlist (id = " ^ test_song_id_skylines ^ ") failed: 404 Not found.")) req_playlist_song_id; 
;;

(* -------- DRIVER --------- *)

let () =
  print_endline "TESTING SPOTIFY MODULE ...";
  test_get_token (); 
  print_endline " get token PASSED";
  test_get_song (); 
  print_endline " get song PASSED";
  test_get_playlist ();
  print_endline " get playlist PASSED";
  test_bad_token_exceptions ();
  print_endline " bad/expired token PASSED";
  test_bad_id_exceptions ();
  print_endline " invalid/ill-formed song and playlist ids PASSED";
;;


