from setuptools import Extension, setup

DIR = "src/match/tensorbase"

tensorbase_module = Extension(
    name="match.tensorbase",
    sources=[
        f"{DIR}/tensorbasemodule.c",
        f"{DIR}/tensorbase_aggregation.c",
        f"{DIR}/tensorbase_alloc.c",
        f"{DIR}/tensorbase_broadcasting.c",
        f"{DIR}/tensorbase_linalg.c",
        f"{DIR}/tensorbase_string.c",
        f"{DIR}/tensorbase_transform.c",
        f"{DIR}/tensorbase_util.c",
    ],
    include_dirs=[DIR],
    extra_compile_args=["-fvisibility=default"],
    extra_link_args=["-undefined", "dynamic_lookup"]
)

setup(ext_modules=[tensorbase_module])
