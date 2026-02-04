{
  description = "Tauri + Rust + OpenCV development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
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
        nativeBuildInputs = with pkgs; [
          (rust-bin.stable.latest.default.override { extensions = [ "rust-src" ]; })
          pkg-config
          cmake
          nodePackages.pnpm
          nodejs_20
          cargo-tauri
          # --- OpenCV/Clangビルドに必須 ---
          clang
          llvmPackages.libclang.lib 
        ];

        buildInputs = libraries;

        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

        # bindgen が標準 C++ ヘッダーを見つけられるようにする
        BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.lib.getVersion pkgs.clang}/include";

        shellHook = ''
          export XDG_DATA_DIRS=${pkgs.gsettings-desktop-schemas}/share/gsettings-schemas/${pkgs.gsettings-desktop-schemas.name}:${pkgs.gtk3}/share/gsettings-schemas/${pkgs.gtk3.name}:$XDG_DATA_DIRS

          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath libraries}:$LD_LIBRARY_PATH
          # shellHook 内でも一応エクスポート
          export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
          
          echo "pnpm + Tauri + OpenCV (with Libclang) environment loaded!"
        '';
      };
    }; 
}
