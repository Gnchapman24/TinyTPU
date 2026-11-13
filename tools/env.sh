# TinyTPU project environment (WSL side). Source it; don't run it:
#
#     source tools/env.sh
#
# It puts only the OSS CAD Suite's bin/ first on PATH, for this shell only.
# It deliberately does NOT use the suite's `environment` script, because that also puts the
# suite's own Python (py3bin/) first on PATH. The project uses Python 3.12 via uv instead.
#
# Override the install location with OSS_CAD_SUITE=/path/to/oss-cad-suite before sourcing.

if [ -z "${BASH_VERSION:-}" ]; then
    echo "tools/env.sh: needs bash" >&2
    return 1 2>/dev/null || exit 1
fi
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "tools/env.sh: source this file instead of running it: source tools/env.sh" >&2
    exit 1
fi

_tinytpu_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=versions.env
. "$_tinytpu_root/tools/versions.env"

OSS_CAD_SUITE="${OSS_CAD_SUITE:-$HOME/tools/oss-cad-suite}"
if [ ! -x "$OSS_CAD_SUITE/bin/yosys" ]; then
    echo "tools/env.sh: OSS CAD Suite not found at $OSS_CAD_SUITE (set OSS_CAD_SUITE)" >&2
    unset _tinytpu_root
    return 1
fi

case ":$PATH:" in
    *":$OSS_CAD_SUITE/bin:"*) ;;
    *) PATH="$OSS_CAD_SUITE/bin:$PATH" ;;
esac
export PATH OSS_CAD_SUITE TINYTPU_ROOT="$_tinytpu_root"

_tinytpu_installed="$(cat "$OSS_CAD_SUITE/VERSION" 2>/dev/null)"
if [ "$_tinytpu_installed" != "$OSS_CAD_SUITE_RELEASE" ]; then
    echo "tools/env.sh: WARNING: OSS CAD Suite release is '$_tinytpu_installed'," \
         "but tools/versions.env pins '$OSS_CAD_SUITE_RELEASE'" >&2
fi
unset _tinytpu_root _tinytpu_installed
