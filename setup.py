from setuptools import Extension, setup

tensorbase_module = Extension(
    name="match.tensorbase",
    sources=[
        "src/match/tensorbase/tensorbasemodule.c",
        "src/match/tensorbase/tensorbase.c",
    ],
    include_dirs=["src/match/tensorbase"],
    extra_compile_args=["-fvisibility=default"],
    extra_link_args=["-undefined", "dynamic_lookup"]
)

setup(ext_modules=[tensorbase_module])
