open Core
open Lwt
open Cohttp
open Cohttp_lwt_unix
open Numpy_helper

let feature_names = [
  "danceability";
  "energy";
  "key";
  "loudness";
  "mode";
  "speechiness";
  "acousticness";
  "instrumentalness";
  "liveness";
  "valence";
  "tempo";
  "duration_ms";
  "time_signature";
]
;;

let spotify_api_credentials = "ZDgwMGZlMzgwOTAwNGRjZGI1NmViZTkwYjg2ZThlNzQ6ZGM0OTdiMWM2MjAwNDY1Y2JlZmYyNWE5ODhkY2YxYzk=";;
let spotify_base_uri = "https://api.spotify.com/v1/";;
let token_request_timeout = 10.;; 
let data_request_timeout = 10.;;


(* --------- HELPERS - HIDDEN ---------- *)

(* --------- GENERAL NETWORK FUNCTIONS ---------- *)

let get_promise_with_timeout ~time ~f =
  Lwt.pick
    [
      (f () >|= fun v -> `Done v)
    ; (Lwt_unix.sleep time >|= fun () -> `Timeout)
    ]

let get_error_msg (body: string): string = 
  body 
  |> Yojson.Safe.from_string 
  |> Yojson.Safe.Util.member "error"
  |> Yojson.Safe.Util.member "message"
  |> Yojson.Safe.to_string
  |> String.filter ~f:(fun c -> Char.(<>) c '"') 

(* ------- ACCESS TOKEN NETWORK FUNCTIONS ------- *)

let send_token_request (): (Response.t * Cohttp_lwt.Body.t) t = 
  let uri = Uri.of_string "https://accounts.spotify.com/api/token?grant_type=client_credentials" in
  let headers = Header.init ()
                |> fun h -> Header.add h "Content-Type" "application/x-www-form-urlencoded"
                            |> fun h -> Header.add h "Authorization" ("Basic " ^ spotify_api_credentials)
  in
  Client.call ~headers `POST uri 

let handle_token_response_get_body (resp: Response.t) (body: Cohttp_lwt.Body.t): string t =
  let code = resp |> Response.status |> Code.code_of_status in
  if code <> 200 then 
    body |> Cohttp_lwt.Body.to_string >>= (function body_str -> 
        Lwt.fail_with @@ "Request for token failed: " ^ (Int.to_string code) ^ " " ^ (get_error_msg body_str))
  else 
    body |> Cohttp_lwt.Body.to_string 

let request_token_return_body () : string t = 
  get_promise_with_timeout ~time:token_request_timeout ~f:send_token_request >>= function
  | `Timeout -> Lwt.fail_with "Request for token failed: timeout expired"
  | `Done (resp, body) -> handle_token_response_get_body resp body 



(* ------- SONG DATA NETWORK FUNCTIONS ------- *) 

let general_api_request (uri_suffix: string) (description: string) (api_token: string): string t =
  let do_request (): (Response.t * Cohttp_lwt.Body.t) t = 
    let uri = Uri.of_string @@ spotify_base_uri ^ uri_suffix in
    let headers = 
      Header.init ()
      |> fun h -> Header.add h "Content-Type" "application/json"
                  |> fun h -> Header.add h "Authorization" ("Bearer " ^ api_token) 
                              |> fun h -> Header.add h "Accept" "application/json"
    in
    Client.call ~headers `GET uri
  in
  let handle_response = fun resp body -> 
    let code = resp |> Response.status |> Code.code_of_status in
    if code <> 200 then
      Cohttp_lwt.Body.to_string body >>= (fun body_str -> 
          Lwt.fail_with @@ description ^ " failed: " ^ (Int.to_string code) ^ " " ^ (get_error_msg body_str)
        )
    else 
      Cohttp_lwt.Body.to_string body                                                                
  in
  get_promise_with_timeout ~f:do_request ~time:data_request_timeout >>= (function 
      | `Timeout -> Lwt.fail_with @@ description ^ " failed: timeout expired" 
      | `Done (resp, body) -> handle_response resp body 
    )

let request_song_metadata (id: string) (api_token: string): string t = 
  let uri_suffix = "tracks/" ^ id in
  let description = "request metadata for song (id = " ^ id ^ ")" in
  general_api_request uri_suffix description api_token

let request_song_features (id: string) (api_token: string): string t = 
  let uri_suffix = "audio-features/" ^ id in
  let description = "request features for song (id = " ^ id ^ ")" in
  general_api_request uri_suffix description api_token 


let request_playlist_metadata (id: string) (api_token: string): string t = 
  let uri_suffix = "playlists/" ^ id in
  let description = "request metadata for playlist (id = " ^ id ^ ")" in
  general_api_request uri_suffix description api_token


let request_song_features_batch (ids_comma_sep: string) (api_token: string): string t = 
  let uri_suffix = "audio-features/?ids=" ^ ids_comma_sep in
  let description = "request batch of song ids" in
  general_api_request uri_suffix description api_token


(* -------- Yojson and (Yojson --> Numpy) data manipulation -------- *)

let yojson_list_remove_nulls (ls: Yojson.Safe.t list): Yojson.Safe.t list = 
  List.filter ~f:(fun elem -> 
      match elem with 
      | `Null -> false 
      | _ -> true) 
    ls

let get_field_remove_quotes (field: string) (obj: string): string = 
  obj 
  |> Yojson.Safe.from_string 
  |> Yojson.Safe.Util.member field
  |> Yojson.Safe.to_string 
  |> String.filter ~f:(fun c -> Char.(<>) c '"') 

let get_audio_feature (features_yojson: Yojson.Safe.t) (field: string): float = 
  features_yojson
  |> Yojson.Safe.Util.member field
  |> Yojson.Safe.to_string 
  |> Float.of_string 

let yojson_extract_list (wrapped_list: Yojson.Safe.t): Yojson.Safe.t list = 
  match wrapped_list with
  | `List ls -> ls |>  List.filter ~f:(fun elem -> match elem with | `Null -> false | _ -> true)
  | _ -> failwith "Yojson object does not have a list at the top level"

let song_features_yojson_to_vector (features_yojson: Yojson.Safe.t): Np.Ndarray.t = 
  List.map feature_names ~f:(get_audio_feature features_yojson) 
  |> Np.Ndarray.of_float_list
  |> Np.reshape ~newshape:[1; (List.length feature_names)]

let playlist_features_yojson_to_matrix (features_yojson: Yojson.Safe.t): Np.Ndarray.t = 
  features_yojson
  |> Yojson.Safe.Util.member "audio_features"
  |> yojson_extract_list 
  |> yojson_list_remove_nulls
  |> List.map ~f:song_features_yojson_to_vector
  |> matrix_of_vector_list

let ids_from_playlist_body (playlist_metadata_body: string): string =
  playlist_metadata_body 
  |> Yojson.Safe.from_string 
  |> Yojson.Safe.Util.member "tracks"
  |> Yojson.Safe.Util.member "items"
  |> yojson_extract_list
  |> List.map ~f:(Yojson.Safe.Util.member "track")
  |> yojson_list_remove_nulls
  |> List.map ~f:(Yojson.Safe.Util.member "id")
  |> List.map ~f:Yojson.Safe.to_string
  |> List.map ~f:(String.filter ~f:(fun c -> Char.(<>) c '"'))
  |> String.concat ~sep:"," 


(* --------- EXPOSED API --------- *)
type song = {name: string; sid: string; features_vector: Np.Ndarray.t;}

type playlist = {name: string; pid: string; features_matrix: Np.Ndarray.t;}


let get_new_api_token _ : string Lwt.t = 
  let%lwt body = request_token_return_body () in  
  let token = get_field_remove_quotes "access_token" body in
  Lwt.return @@ token 

let playlist_of_id (pid: string) (api_token: string): playlist Lwt.t =
  let%lwt playlist_body = request_playlist_metadata pid api_token in
  let name = get_field_remove_quotes "name" playlist_body in
  let ids = ids_from_playlist_body playlist_body in
  let%lwt batch_body = request_song_features_batch ids api_token in
  let features_matrix = batch_body |> Yojson.Safe.from_string |> playlist_features_yojson_to_matrix in
  Lwt.return {name; pid; features_matrix;}

let save_playlist (p: playlist) (file: string) : unit =
  let f = Stdio.Out_channel.create file
  in Stdio.Out_channel.output_string f 
    @@ p.name ^ "\n" ^ p.pid ^ (matrix_to_string p.features_matrix);
  Stdio.Out_channel.flush f;
  Stdio.Out_channel.close f

let load_playlist (file: string) : playlist =
  match Stdio.In_channel.read_lines file with 
  | name :: pid :: matrix 
    ->  {features_matrix = txt_to_matrix matrix; name; pid;}
  | _ -> failwith "improper file formatting"

let replace_features (p: playlist) (f: Np.Ndarray.t) : playlist =
  {features_matrix = f; name = p.name; pid = p.pid}

let song_of_id (sid: string) (api_token: string): song Lwt.t =
  let%lwt features_body = request_song_features sid api_token in
  let%lwt metadata_body = request_song_metadata sid api_token in
  let name = get_field_remove_quotes "name" metadata_body in
  let features_body_yojson = Yojson.Safe.from_string features_body in 
  (* fail if this song doesn't have features *)
  match features_body_yojson with 
  |`Null -> Lwt.fail_with @@  "song (id = " ^ sid ^ ") does not have audio feature data"
  | _ -> 
    let features_vector = song_features_yojson_to_vector features_body_yojson in
    Lwt.return {name; sid; features_vector;}