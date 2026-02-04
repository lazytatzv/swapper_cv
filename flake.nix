{
  description = "Tauri + Rust + OpenCV development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      # system のハイフンをアンダースコアに修正 (x86_64-linux)
      system = "x86_64-linux"; 

      pkgs = import nixpkgs {
        inherit system;
        overlays = [ (import rust-overlay) ];
      };

      libraries = with pkgs; [
        webkitgtk_4_1
        gtk3
        cairo
        pango
        atk
        gdk-pixbuf
        libsoup_3
        opencv
        stdenv.cc.cc.lib
      ];
    
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (rust-bin.stable.latest.default.override { extensions = [ "rust-src" ]; })
          pkg-config
          cmake
          nodePackages.pnpm
          nodejs_20
          cargo-tauri
        ] ++ libraries;

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath libraries}:$LD_LIBRARY_PATH
          echo "pnpm + Tauri + OpenCV environment loaded!"
        '';
      };
    }; 
}
