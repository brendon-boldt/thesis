{
  outputs = { self, nixpkgs }:
  let 
    system =  "x86_64-linux";
    pkgs = import nixpkgs { system = system; };
    python-with-packages = pkgs.python312.withPackages (p: with p; [ ]);
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        graphviz
        texlivePackages.latexmk
      ];
    };
  };
}

