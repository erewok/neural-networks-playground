_default:
    @just --list

# run one exercise by number, e.g. `just run 1`. `just run all` runs every one.
run N *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{N}}" = "all" ]; then
        for f in 0*_*.py; do
            printf '\n\033[1;36m=== %s ===\033[0m\n' "$f"
            uv run "$f"
        done
    else
        uv run 0{{N}}_*.py {{ARGS}}
    fi

# same as `just run all`
all: (run "all")

# open the figures for one exercise, e.g. `just plot 1`
plot N:
    @uv run 0{{N}}_*.py plot

# write the figures to PNGs instead of opening windows
save-plots N dir="figures":
    @NN_PLOT_SAVE={{dir}} uv run 0{{N}}_*.py plot
