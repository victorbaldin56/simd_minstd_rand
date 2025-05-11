include(default)

[settings]
compiler=clang
compiler.version=19
build_type=Debug

[buildenv]
CC=clang
CXX=clang++

[conf]
tools.build:cflags=["-fsanitize=address,leak,undefined"]
tools.build:cxxflags=["-fsanitize=address,leak,undefined"]
tools.build:exelinkflags=["-fsanitize=address,leak,undefined"]
tools.build:sharedlinkflags=["-fsanitize=address,leak,undefined"]
