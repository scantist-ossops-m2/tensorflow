"""build_defs for service/cpu."""

def runtime_copts():
    """Returns copts used for CPU runtime libraries."""
    return (["-DEIGEN_AVOID_STL_ARRAY"] + select({
        "//xla/tsl:android_arm": ["-mfpu=neon"],
        "//conditions:default": [],
    }) + select({
        "//xla/tsl:android": ["-O2"],
        "//conditions:default": [],
    }))
