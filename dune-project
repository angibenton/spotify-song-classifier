(lang dune 2.9)
(generate_opam_files true)
(authors "Angi Benton" "Jack Van Holland")
(maintainers "jvanhol1@jhu.edu")
(package
 (name spotify-song-classifier) ; put in your own project name here
 (synopsis "A machine learning CLI application for classifying Spotify songs")
 (description "Can ")
 (depends
  (ounit2 (= 2.2.4)) 
  (ounit2-lwt (= 2.2.4)) 
  (core (= 0.14.1)) 
  (yojson (= 1.7.0))
  (sklearn (= 0.23-0.3.1))
  (ppx_let (= 0.14.0))
  (ppx_deriving_yojson (= 3.6.1))
  (lwt_ppx (= 2.0.2))
  (lwt (= 5.4.2))
  (cohttp-lwt-unix (= 4.0.0))
))