with import <nixpkgs> {};

stdenv.mkDerivation {
    name = "poetry";
    buildInputs = [
      fish
      git
      python311
      python311Packages.black
      python311Packages.ipython
      poetry
      ruff

      gitRepo
      gnupg
      autoconf
      curl
      procps
      gnumake
      util-linux
      m4
      gperf
      unzip
      cudatoolkit
      linuxPackages.nvidia_x11
      libGLU libGL
      xorg.libXi xorg.libXmu freeglut
      xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib
      ncurses5
      stdenv.cc
      binutils

    ];
    shellHook = ''
        export ENVNAME=poetry
        export CUDA_PATH=${pkgs.cudatoolkit}
        # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
        fish --init-command="function fish_greeting; echo 'Entered $ENVNAME Environment'; end; function fish_prompt; echo '$ENVNAME 🐟> '; end;"
        echo "Leaving $ENVNAME Environment"
        exit
    '';
}
