_default:
    @just --list

# Exercises live in chapter directories but keep one global numbering, so
# `just run 7` finds exercise 7 without you having to remember its chapter.
#
# harness.py and plots.py stay at the root and every exercise imports them
# plainly, so PYTHONPATH points there rather than each file carrying a
# sys.path shim at the top of its docstring.
export PYTHONPATH := justfile_directory()

# run one exercise by number, e.g. `just run 5`. also `just run all` and
# `just run chapter 2`. extra args are passed through, e.g. `just run 3 seeds 100`
run N *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{N}}" = "chapter" ]; then
        exec just chapter {{ARGS}}
    fi
    if [ "{{N}}" = "all" ]; then
        for f in ch*/[0-9][0-9]_*.py; do
            printf '\n\033[1;36m=== %s ===\033[0m\n' "$f"
            uv run "$f"
        done
        exit 0
    fi
    f=$(ls ch*/$(printf '%02d' "{{N}}")_*.py 2>/dev/null | head -1 || true)
    if [ -z "$f" ]; then
        echo "no exercise {{N}} yet. \`just chapters\` shows what exists."
        exit 1
    fi
    uv run "$f" {{ARGS}}

# run every exercise in one chapter, e.g. `just chapter 1`
chapter N:
    #!/usr/bin/env bash
    set -euo pipefail
    dir=$(ls -d ch{{N}}_* 2>/dev/null | head -1 || true)
    if [ -z "$dir" ]; then
        echo "no chapter {{N}}. \`just chapters\` shows what exists."
        exit 1
    fi
    shopt -s nullglob
    files=("$dir"/[0-9][0-9]_*.py)
    if [ ${#files[@]} -eq 0 ]; then
        echo "chapter {{N}} ($dir) is empty -- nothing written yet."
        exit 0
    fi
    printf '\n\033[1;35m##### %s #####\033[0m\n' "$dir"
    for f in "${files[@]}"; do
        printf '\n\033[1;36m=== %s ===\033[0m\n' "$(basename "$f")"
        uv run "$f"
    done

# what exists, chapter by chapter
chapters:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    for dir in ch*/; do
        dir=${dir%/}
        n=${dir%%_*}
        printf '\n\033[1;35m%s\033[0m  \033[2m(just chapter %s)\033[0m\n' \
            "$dir" "${n#ch}"
        files=("$dir"/[0-9][0-9]_*.py)
        if [ ${#files[@]} -eq 0 ]; then
            printf '  \033[2mnothing written yet\033[0m\n'
            continue
        fi
        for f in "${files[@]}"; do
            base=$(basename "$f")
            printf '  \033[36m%-28s\033[0m \033[2mjust run %s\033[0m\n' \
                "$base" "$((10#${base%%_*}))"
        done
    done

# same as `just run all`
all: (run "all")

# open the figures for one exercise, e.g. `just plot 1`
plot N:
    #!/usr/bin/env bash
    set -euo pipefail
    f=$(ls ch*/$(printf '%02d' "{{N}}")_*.py 2>/dev/null | head -1 || true)
    if [ -z "$f" ]; then echo "no exercise {{N}} yet."; exit 1; fi
    uv run "$f" plot

# write the figures to PNGs instead of opening windows
save-plots N dir="figures":
    #!/usr/bin/env bash
    set -euo pipefail
    f=$(ls ch*/$(printf '%02d' "{{N}}")_*.py 2>/dev/null | head -1 || true)
    if [ -z "$f" ]; then echo "no exercise {{N}} yet."; exit 1; fi
    NN_PLOT_SAVE={{dir}} uv run "$f" plot
