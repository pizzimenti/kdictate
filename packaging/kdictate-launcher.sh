#!/bin/sh
# Thin launcher for kdictate's console entry points in a packaged install.
#
# kdictate's runtime deps are vendored under /usr/lib/kdictate/vendor (see
# PKGBUILD), not in the system site-packages, so the wheel-generated entry
# scripts can't find them on their own. This launcher prepends the vendor
# dir (plus the system site-packages, where the kdictate package itself
# lives) to PYTHONPATH, then execs the real entry point of the same name
# from libexec.
#
# Installed as /usr/bin/{kdictate-daemon,kdictatectl,ibus-engine-kdictate},
# each a symlink to this script; basename "$0" selects which entry to run.
set -eu

VENDOR_DIR="/usr/lib/kdictate/vendor"
SITE="$(/usr/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONPATH="${VENDOR_DIR}:${SITE}${PYTHONPATH:+:$PYTHONPATH}"

exec /usr/lib/kdictate/libexec/"$(basename "$0")" "$@"
