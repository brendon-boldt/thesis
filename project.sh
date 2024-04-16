#!/usr/bin/env zsh

self=$0
maintex=src/main.tex

setopt globdots nullglob

function watch {
  mkdir -p output/
  while true; do
      inotifywait -qqe modify -e move_self **/*.{tex,bib}
      echo -n "Compiling...  "
      export max_print_line=1048576
      latexmk \
          -f \
          -pdf \
          -interaction=nonstopmode \
          -outdir=output/ \
          -jobname=document \
          $maintex >output/stdout
      echo Done. "($?)"
  done
}

function show {
  &>/dev/null zathura --fork output/document.pdf
}

function setup {
  tmux rename-window "$(basename $(dirname $(realpath $self)))"
  tmux send-keys "nvim" Enter
  tmux split-pane -h
  tmux send-keys "$self watch" Enter
  tmux split-pane -v
  tmux resize-pane -y 80%
  tmux send-keys "$self show; git fetch" Enter
}

"$@"
