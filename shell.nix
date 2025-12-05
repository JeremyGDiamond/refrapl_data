{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.pandas
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.scipy
    pkgs.python3Packages.seaborn
    pkgs.python3Packages.jinja2
  ];

  shellHook = ''
    echo "Python environment ready. Using: $(python --version)"
  '';
}
