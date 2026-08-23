_default:
    @just --list

# run one exercise by number, e.g. `just run 1` (extra args pass through)
run N *ARGS:
    @uv run 0{{N}}_*.py {{ARGS}}

# run all of them
all:
    @for f in 0*_*.py; do echo "=== $f ==="; uv run "$f"; echo; done
