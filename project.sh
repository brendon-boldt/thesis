#!/usr/bin/env zsh

self=$0
maintex=src/main.tex

setopt globdots nullglob

function watch {
  while true; do
      inotifywait -qqe modify -e move_self **/*.{tex,bib,dot}
      sleep 0.1  # Prevents a race condition between how nvim saves files and Make
      mkdir -p output/
      echo -n "Compiling...  "
      export max_print_line=19999
      >output/stdout make
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
