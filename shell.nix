{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.pandas
    pkgs.python3Packages.matplotlib
  ];

  shellHook = ''
    echo "Python environment ready. Using: $(python --version)"
  '';
}
